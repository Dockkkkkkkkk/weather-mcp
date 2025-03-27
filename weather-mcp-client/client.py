#!/usr/bin/env python3

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import httpx
from mcp import ClientSession, Tool, stdio_client, StdioServerParameters
from openai import OpenAI
from contextlib import AsyncExitStack

# 添加dotenv支持，从.env文件加载环境变量
try:
    from dotenv import load_dotenv
    # 尝试从当前目录和项目根目录加载.env文件
    load_dotenv()
    print("已从.env文件加载环境变量配置")
except ImportError:
    print("提示: 安装 python-dotenv 可以从.env文件加载配置 (pip install python-dotenv)")

# 通义千问API相关配置
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
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

class MCPClient:
    """通用MCP客户端，可连接任何MCP服务器，并可选集成大模型API"""
    
    def __init__(self, config_path="mcp-servers.json"):
        """初始化客户端
        
        Args:
            config_path: MCP服务器配置文件路径
            llm_provider: 大模型提供商，目前支持"tongyiqianwen"或None（不使用大模型）
        """
        self.config_path = config_path
        self.servers = {}  # 服务器名称到连接的映射
        self.sessions = {}  # 服务器名称到会话的映射
        self.tools_by_server = {}  # 服务器名称到工具列表的映射
        self.all_tools = []  # 所有可用工具的列表
        self.exit_stack = AsyncExitStack()  # 用于管理异步资源的退出栈
        
        # 初始化大模型客户端
        if LLM_API_KEY:
            self.llm_client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL
            )
        else:
            self.llm_client = None
    
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
            
            print(tools_result)
            tools = tools_result.tools
            
            print(tools)
            
            # 存储工具信息
            self.tools_by_server[server_name] = tools
            self.all_tools.extend([(tool, server_name) for tool in tools])
            
            print(f"已连接到服务器 {server_name}，可用工具: {[tool.name for tool in tools]}")
            return True
            
        except Exception as e:
            print(f"连接到服务器 {server_name} 时出错: {str(e)}")
            return False
        
    async def initialize(self):
        """初始化MCP客户端，连接到所有配置的服务器"""
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
    
    def find_tool(self, tool_name: str) -> Tuple[Optional[Tool], Optional[str]]:
        """查找指定名称的工具及其所属服务器
        
        Args:
            tool_name: 工具名称
            
        Returns:
            (Tool, 服务器名称) 或 (None, None) 如果未找到
        """
        for tool, server_name in self.all_tools:
            if tool.name == tool_name:
                return tool, server_name
        return None, None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用MCP服务器工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具调用结果
        """
        tool, server_name = self.find_tool(tool_name)
        if not tool or not server_name:
            print(f"错误：未找到名为 {tool_name} 的工具")
            return f"错误：未找到名为 {tool_name} 的工具"
        
        session = self.sessions.get(server_name)
        if not session:
            print(f"错误：未找到名为 {server_name} 的服务器会话")
            return f"错误：未找到名为 {server_name} 的服务器会话"
        
        try:
            result = await session.call_tool(tool_name, arguments)
            content = result.content[0]
            print(f"工具 {tool_name} 返回结果: {content.text}")
            return content.text
        except Exception as e:
            print(f"调用工具 {tool_name} 时出错: {str(e)}")
            return f"调用工具 {tool_name} 时出错: {str(e)}"
    
    async def query_llm(self, user_query: str, context: Optional[str] = None) -> str:
        """查询大模型
        
        Args:
            user_query: 用户查询
            context: 上下文信息
            
        Returns:
            大模型的回复
        """
        if not self.llm_client:
            tools_info = "\n".join([f"- {tool.name}: {tool.description}" for tool, _ in self.all_tools])
            return f"错误：未设置LLM_API_KEY环境变量，无法访问大模型API。\n可用的工具有：\n{tools_info}"
        
        try:
            # 构建消息内容
            messages = [
                {"role": "system", "content": "你是一个智能助手，可以通过各种工具帮助用户。"}
            ]
            
            # 如果有上下文，加入到系统消息中
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"以下是相关信息，请根据这些信息回答用户的问题：\n\n{context}"
                })
            
            # 添加用户问题
            messages.append({"role": "user", "content": user_query})
            
            # 格式化工具信息，符合OpenAI标准
            formatted_tools = []
            for tool, server_name in self.all_tools:
                # 将MCP工具转换为OpenAI工具格式
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                formatted_tools.append(formatted_tool)
            
            # 使用OpenAI SDK调用大模型
            completion = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                tool_choice="auto"  # 让模型自动决定是否使用工具
            )
            
            # 提取回复内容
            response = completion.choices[0].message
            
            # 处理工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 调用对应的MCP工具
                    tool_result = await self.call_tool(function_name, function_args)
                    tool_results.append(f"工具 {function_name} 返回结果: {tool_result}")
                
                # 将工具结果作为上下文，再次调用大模型进行总结
                tool_results_text = "\n".join(tool_results)
                followup_messages = messages + [
                    {"role": "assistant", "content": response.content if response.content else ""},
                    {"role": "system", "content": f"工具返回的结果:\n{tool_results_text}"},
                    {"role": "user", "content": "根据工具返回的结果，请给出完整回答"}
                ]
                
                followup_completion = self.llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=followup_messages
                )
                
                final_reply = followup_completion.choices[0].message.content
                return final_reply
            else:
                # 如果没有工具调用，直接返回模型回复
                return response.content
                
        except Exception as e:
            return f"与大模型API通信时出错: {str(e)}"
    
    async def process_query(self, query: str) -> str:
        """处理用户查询，根据需要调用工具并集成大模型
        
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
            
            # 尝试解析参数
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
            return await self.call_tool(tool_name, arguments)
        
        # 如果配置了大模型，则使用大模型进行处理
        if self.llm_client:
            return await self.query_llm(query)
        
        # 如果没有配置大模型，显示帮助信息
        tools_info = "\n".join([f"- /{tool.name}: {tool.description}" for tool, _ in self.all_tools])
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
    
    client = MCPClient(config_path)
    try:
        await client.initialize()
        
        # 检查API密钥是否设置
        if client.llm_client:
            print("\n✅ 大模型API已连接，可以直接输入自然语言问题")
        else:
            print("\n⚠️ 警告：未设置LLM_API_KEY环境变量，无法使用大模型功能")
            print("请通过以下方式设置API密钥：")
            print("方法1: 环境变量设置:")
            print("  Windows: $env:LLM_API_KEY=\"你的API密钥\"")
            print("  Linux/Mac: export LLM_API_KEY=\"你的API密钥\"")
            print("方法2: 创建.env文件:")
            print("  复制.env.example为.env并填写API密钥")
            print("\n目前只能使用命令行方式调用工具: /<工具名> <参数>\n")
        
        print("欢迎使用MCP客户端 (输入'退出'结束对话，输入'/帮助'获取帮助)")
        print("---------------------------------------------")
        print("可用命令格式: /<工具名> [参数]")
        print("例如: /get_weather city=北京")
        if client.llm_client:
            print("或者直接输入自然语言问题（如：北京今天天气怎么样？）")
        print("---------------------------------------------")
        
        while True:
            user_input = input("\n请输入您的问题或命令: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                break
                
            if user_input.lower() in ["/帮助", "/help"]:
                tools_info = "\n".join([f"- /{tool.name}: {tool.description}" for tool, _ in client.all_tools])
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