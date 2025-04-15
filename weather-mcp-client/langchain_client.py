#!/usr/bin/env python3

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from mcp import ClientSession, Tool, stdio_client, StdioServerParameters
from contextlib import AsyncExitStack
from pydantic import BaseModel, Field

# 导入LangChain相关库
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.base import BaseTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

# 添加dotenv支持，从.env文件加载环境变量
try:
    from dotenv import load_dotenv
    # 尝试从当前目录和项目根目录加载.env文件
    load_dotenv()
    print("已从.env文件加载环境变量配置")
except ImportError:
    print("提示: 安装 python-dotenv 可以从.env文件加载配置 (pip install python-dotenv)")

# 通义千问API相关配置
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-4aee53c9947b4f369b3a0f9ba2e7cbbc")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-plus")

# 打印环境变量状态，帮助调试
if LLM_API_KEY:
    # 不显示完整API密钥，只显示前6位和后4位
    masked_key = LLM_API_KEY[:6] + "*" * (len(LLM_API_KEY) - 10) + LLM_API_KEY[-4:] if len(LLM_API_KEY) > 10 else "***"
    print(f"已检测到LLM_API_KEY: {masked_key}")
    print(f"LLM_BASE_URL: {LLM_BASE_URL}")
    print(f"MODEL_NAME: {MODEL_NAME}")
else:
    print("警告: 未设置LLM_API_KEY环境变量，无法使用大模型功能")


def fix_schema_compatibility(schema):
    """
    修复JSON Schema兼容性问题，处理oneOf/allOf/anyOf结构
    
    Args:
        schema: 原始schema字典
        
    Returns:
        修复后的schema字典
    """
    if not isinstance(schema, dict):
        return schema
    
    # 删除顶层的oneOf/allOf/anyOf
    for key in ['oneOf', 'allOf', 'anyOf']:
        if key in schema:
            # 如果是顶层，尝试合并或使用第一个选项
            if key == 'oneOf' or key == 'anyOf':
                if schema[key] and isinstance(schema[key], list) and len(schema[key]) > 0:
                    # 使用第一个选项
                    first_option = schema[key][0]
                    schema.pop(key)
                    if isinstance(first_option, dict):
                        for k, v in first_option.items():
                            if k not in schema:
                                schema[k] = v
            elif key == 'allOf':
                if schema[key] and isinstance(schema[key], list):
                    merged = {}
                    for item in schema[key]:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                merged[k] = v
                    schema.pop(key)
                    for k, v in merged.items():
                        schema[k] = v
    
    # 处理嵌套的属性
    for key, value in list(schema.items()):
        if isinstance(value, dict):
            schema[key] = fix_schema_compatibility(value)
        elif isinstance(value, list):
            schema[key] = [fix_schema_compatibility(item) if isinstance(item, dict) else item for item in value]
    
    return schema


class MCPToolWrapper(BaseTool):
    """包装MCP工具为LangChain工具"""
    
    name: str = ""
    description: str = ""
    session: Any = None
    server_name: str = ""
    
    def _run(self, **kwargs) -> str:
        """同步运行工具"""
        raise NotImplementedError("请使用_arun方法")
    
    async def _arun(self, **kwargs) -> str:
        """异步运行工具"""
        tool_name = self.name
        try:
            print(f"调用工具 {tool_name} 参数: {kwargs}")
            result = await self.session.call_tool(tool_name, kwargs)
            content = result.content[0] if hasattr(result, 'content') else result.contents[0]
            response = content.text
            print(f"工具 {tool_name} 返回结果: {response}")
            return response
        except Exception as e:
            error_msg = f"调用工具 {tool_name} 时出错: {str(e)}"
            print(error_msg)
            return error_msg


class LangChainMCPClient:
    """基于LangChain的MCP客户端，支持会话上下文管理和多轮函数调用"""
    
    def __init__(self, config_path="mcp-servers.json"):
        """初始化客户端
        
        Args:
            config_path: MCP服务器配置文件路径
        """
        self.config_path = config_path
        self.servers = {}  # 服务器名称到连接的映射
        self.sessions = {}  # 服务器名称到会话的映射
        self.tools_by_server = {}  # 服务器名称到工具列表的映射
        self.all_tools = []  # 所有可用工具的列表
        self.exit_stack = AsyncExitStack()  # 用于管理异步资源的退出栈
        self.langchain_tools = []  # LangChain工具列表
        self.agent_executor = None  # LangChain代理执行器
        self.memory = None  # 对话记忆
        
        # 初始化大模型客户端
        if LLM_API_KEY:
            self.llm = ChatOpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                model=MODEL_NAME,
                streaming=True,
                temperature=0
            )
        else:
            self.llm = None
    
    def load_server_configs(self) -> Dict[str, Dict]:
        """从配置文件加载所有服务器配置
        
        Returns:
            服务器名称到配置的映射
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if 'mcpServers' not in config:
                print(f"错误：配置文件 {self.config_path} 中未找到有效的服务器配置")
                return {}
                
            return config['mcpServers']
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"加载配置文件时出错: {str(e)}")
            return {}
    
    async def connect_to_server(self, server_name: str, server_config: Dict) -> bool:
        """连接到指定的MCP服务器
        
        Args:
            server_name: 服务器名称
            server_config: 服务器配置
            
        Returns:
            连接是否成功
        """
        try:
            # 提取服务器启动命令、参数和环境变量
            command = server_config.get('command', 'python')
            args = server_config.get('args', [])
            env = server_config.get('env', {})
            
            # 合并环境变量
            merged_env = os.environ.copy()
            for key, value in env.items():
                merged_env[key] = value
            
            # Windows平台特殊处理
            if os.name == 'nt' and command == 'npx':
                command = 'cmd'
                args = ['/c', 'npx'] + args
                print(f"Windows平台检测到npx命令，已转换为: {command} {' '.join(args)}")
            
            # 构建服务器参数
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=merged_env
            )
            
            print(f"开始连接到服务器 {server_name}...")
            
            # 启动服务器并连接
            transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            # 解包transport获取read_stream和write_stream
            read_stream, write_stream = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            
            # 保存会话
            self.sessions[server_name] = session
            
            # 初始化连接
            await session.initialize()
            
            # 获取可用工具列表
            print(f"正在获取 {server_name} 服务器的工具列表...")
            tools_result = await session.list_tools()
            
            tools = tools_result.tools
            
            # 存储工具信息
            self.tools_by_server[server_name] = tools
            self.all_tools.extend([(tool, server_name) for tool in tools])
            
            # 创建LangChain工具
            for tool in tools:
                # 修复工具的input_schema以确保兼容性
                fixed_schema = None
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    try:
                        # 深拷贝schema以避免修改原始对象
                        schema_copy = json.loads(json.dumps(tool.inputSchema))
                        fixed_schema = fix_schema_compatibility(schema_copy)
                        print(f"工具 {tool.name} schema已修复")
                    except Exception as e:
                        print(f"修复工具 {tool.name} schema时出错: {str(e)}")
                        fixed_schema = tool.inputSchema
                
                langchain_tool = MCPToolWrapper(
                    name=tool.name,
                    description=tool.description,
                    session=session,
                    server_name=server_name
                )
                
                # 修改工具的schema属性
                if fixed_schema:
                    setattr(langchain_tool, 'args_schema', fixed_schema)
                
                self.langchain_tools.append(langchain_tool)
            
            print(f"已连接到服务器 {server_name}，可用工具: {[tool.name for tool in tools]}")
            return True
            
        except Exception as e:
            print(f"连接到服务器 {server_name} 时出错: {str(e)}")
            return False
        
    async def initialize(self):
        """初始化MCP客户端，连接到所有配置的服务器并创建LangChain Agent"""
        server_configs = self.load_server_configs()
        if not server_configs:
            raise ValueError(f"无法从配置文件 {self.config_path} 加载服务器配置")
        
        # 连接到所有服务器
        connection_results = []
        for server_name, server_config in server_configs.items():
            result = await self.connect_to_server(server_name, server_config)
            connection_results.append((server_name, result))
        
        # 检查是否至少有一个服务器连接成功
        if not any(result for _, result in connection_results):
            failed_servers = [name for name, result in connection_results if not result]
            raise ValueError(f"无法连接到任何MCP服务器: {', '.join(failed_servers)}")
            
        print(f"已连接到 {sum(1 for _, result in connection_results if result)}/{len(connection_results)} 个MCP服务器")
        
        # 如果大模型配置正确且有工具，则创建Agent
        if self.llm and self.langchain_tools:
            self._setup_agent()
            print("✅ 已创建LangChain Agent，支持多轮对话和工具调用")
    
    def _setup_agent(self):
        """设置LangChain Agent"""
        # 创建对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )
        
        # 创建代理提示模板
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是一个智能助手，可以通过各种工具帮助用户。"
                "分析用户的问题，并使用提供的工具来解决问题。"
                "工具使用过程中，先思考使用哪个工具和参数，然后再执行。"
                "对于不需要工具的问题，直接回答用户。"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建代理
        agent = create_openai_tools_agent(self.llm, self.langchain_tools, prompt)
        
        # 创建代理执行器
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10  # 允许多轮工具调用
        )
    
    async def process_query(self, query: str) -> str:
        """处理用户查询，使用LangChain Agent
        
        Args:
            query: 用户查询文本
            
        Returns:
            处理结果
        """
        if query.startswith('/'):
            # 命令模式，直接调用工具
            parts = query[1:].split(maxsplit=1)
            if len(parts) == 0:
                return "请指定要调用的工具"
            
            tool_name = parts[0]
            args_text = parts[1] if len(parts) > 1 else ""
            
            # 查找对应的LangChain工具
            tool = next((t for t in self.langchain_tools if t.name == tool_name), None)
            if not tool:
                return f"错误：未找到名为 {tool_name} 的工具"
            
            # 解析参数
            try:
                arguments = {}
                if args_text:
                    # 支持两种形式: "key=value key2=value2" 或 JSON格式 "{key: value, key2: value2}"
                    if args_text.strip().startswith('{'):
                        arguments = json.loads(args_text)
                    else:
                        for arg in args_text.split():
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                arguments[key] = value
            except Exception as e:
                return f"解析工具参数时出错: {str(e)}"
            
            # 调用工具
            response = await tool._arun(**arguments)
            return response
        
        # 使用LangChain Agent进行自然语言处理
        if self.agent_executor:
            try:
                # 异步调用Agent
                result = await asyncio.to_thread(self.agent_executor.invoke, {"input": query})
                return result["output"]
            except Exception as e:
                return f"处理查询时出错: {str(e)}"
        else:
            # 如果Agent未初始化，显示帮助信息
            tools_info = "\n".join([f"- /{tool.name}: {tool.description}" for tool in self.langchain_tools])
            return f"请使用 /<工具名> <参数> 的格式调用工具，例如 '/get_weather city=北京'\n\n可用的工具有：\n{tools_info}"
    
    async def close(self):
        """关闭所有客户端会话和连接"""
        try:
            # 使用退出栈自动清理所有资源
            await self.exit_stack.aclose()
            print("已关闭所有连接")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")


async def main():
    """主函数"""
    # 允许从命令行指定配置文件路径
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "mcp-servers.json"
    
    # 打印当前路径和所查找的配置文件，帮助调试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试加载配置文件: {config_path}")
    
    client = LangChainMCPClient(config_path)
    try:
        await client.initialize()
        
        # 检查是否初始化成功
        if client.agent_executor:
            print("\n✅ LangChain Agent已初始化，支持多轮对话和工具调用")
        else:
            print("\n⚠️ 警告：未设置LLM_API_KEY环境变量或无法连接到MCP服务器，无法使用Agent功能")
            print("请通过以下方式设置API密钥：")
            print("方法1: 环境变量设置:")
            print("  Windows: $env:LLM_API_KEY=\"你的API密钥\"")
            print("  Linux/Mac: export LLM_API_KEY=\"你的API密钥\"")
            print("方法2: 创建.env文件:")
            print("  复制.env.example为.env并填写API密钥")
            print("\n目前只能使用命令行方式调用工具: /<工具名> <参数>\n")
        
        print("欢迎使用LangChain MCP客户端 (输入'退出'结束对话，输入'/帮助'获取帮助)")
        print("---------------------------------------------")
        print("可用命令格式: /<工具名> [参数]")
        print("例如: /get_weather city=北京")
        if client.agent_executor:
            print("或者直接输入自然语言问题（如：北京今天天气怎么样？）")
            print("支持多轮对话和连续工具调用")
        print("---------------------------------------------")
        
        while True:
            user_input = input("\n请输入您的问题或命令: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                break
                
            if user_input.lower() in ["/帮助", "/help"]:
                tools_info = "\n".join([f"- /{tool.name}: {tool.description}" for tool in client.langchain_tools])
                print(f"\n可用的工具有：\n{tools_info}")
                continue
                
            print("\n正在处理您的请求...")
            response = await client.process_query(user_input)
            print(f"\n回答: {response}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main()) 