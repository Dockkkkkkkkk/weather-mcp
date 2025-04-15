#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangChain MCP适配器示例脚本 - 连接common-api-server

这个脚本展示了如何使用langchain-mcp-adapters库连接MCP服务器，
加载工具，并使用LangChain Agent处理用户查询。
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# LangChain核心组件
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough

# LangChain组件
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

# OpenAI集成
from langchain_openai import ChatOpenAI

# MCP适配器
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LangChainMCPAPIExample:
    """
    使用LangChain适配器连接common-api-server的示例类
    
    这个类展示了如何:
    1. 连接到common-api-server MCP服务器
    2. 加载MCP工具
    3. 将MCP工具转换为LangChain工具
    4. 创建LangChain Agent
    5. 处理用户查询
    """
    
    def __init__(self, config_path: str = "mcp-servers.json"):
        """
        初始化LangChain MCP示例
        
        Args:
            config_path: 配置文件路径，默认为mcp-servers.json
        """
        self.config_path = config_path
        self.server_configs = {}
        self.mcp_client = None
        self.tools = []
        self.agent_executor = None
        self.memory = []  # 保存对话历史
        
    async def load_server_configs(self) -> Dict:
        """
        加载服务器配置
        
        Returns:
            包含服务器配置的字典
        """
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # 提取mcpServers部分
                self.server_configs = json_data.get("mcpServers", {})
                logger.info(f"已加载MCP服务器配置: {self.config_path}")
                return self.server_configs
        except Exception as e:
            logger.error(f"加载服务器配置时发生错误: {str(e)}")
            raise
    
    def setup_agent(self, model_name: str = None, temperature: float = 0, enable_tools: bool = True) -> AgentExecutor:
        """
        设置LangChain Agent
        
        Args:
            model_name: 要使用的模型名称，如果为None则从环境变量读取
            temperature: 模型温度参数
            enable_tools: 是否启用工具，设为False时将不使用工具（纯聊天模式）
            
        Returns:
            配置好的AgentExecutor
        """
        if not self.tools and enable_tools:
            raise ValueError("请先加载工具再设置Agent")
            
        # 从环境变量获取模型配置
        model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 创建LLM
        llm_params = {
            "model": model_name,
            "temperature": temperature,
        }
        
        # 如果设置了base_url，添加到参数中
        if base_url:
            llm_params["base_url"] = base_url
            
        # 如果设置了api_key，添加到参数中
        if api_key:
            llm_params["api_key"] = api_key
            
        # 输出使用的模型信息（不包括敏感信息）
        logger.info(f"使用模型: {model_name}")
        if base_url:
            logger.info(f"使用自定义API地址: {base_url}")
        
        # 创建LLM实例
        llm = ChatOpenAI(**llm_params)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能助手，能够利用各种API工具帮助用户解决问题。
            你可以处理数据，发送请求，管理字典数据和处理API文档等任务。
            请仔细分析用户的需求，选择最合适的工具来完成任务，并将结果以清晰易懂的方式呈现给用户。
            在处理敏感信息时，请注意保密性。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 如果不启用工具，则使用纯聊天模式
        if not enable_tools:
            logger.info("工具功能已禁用，使用纯聊天模式")
            
            # 简单的聊天链
            chat_chain = prompt | llm
            
            # 创建一个简单的执行器，模拟AgentExecutor的接口
            class SimpleChatExecutor:
                async def ainvoke(self, inputs):
                    response = await chat_chain.ainvoke(inputs)
                    return {"output": response.content}
            
            self.agent_executor = SimpleChatExecutor()
            logger.info("已成功设置纯聊天模式")
            return self.agent_executor
        
        try:
            # 适配DashScope API的函数格式
            # 将工具元数据转换为兼容的格式
            tools_for_binding = []
            
            logger.info(f"开始处理 {len(self.tools)} 个工具:")
            
            for i, tool in enumerate(self.tools):
                try:
                    # 检查工具是否有metadata属性
                    if not hasattr(tool, 'metadata') or tool.metadata is None:
                        logger.warning(f"工具 {i+1}/{len(self.tools)} - 缺少metadata属性，尝试从工具对象中提取信息")
                        # 尝试直接从工具对象获取必要信息
                        tool_metadata = {
                            "name": getattr(tool, 'name', f"tool_{i}"),
                            "description": getattr(tool, 'description', f"工具 {i+1}"),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    else:
                        tool_metadata = tool.metadata
                    
                    logger.info(f"工具 {i+1}/{len(self.tools)} - 元数据: {tool_metadata.get('name', getattr(tool, 'name', f'tool_{i}'))}")
                    
                    # 构建格式化的工具
                    formatted_tool = {
                        "name": tool_metadata.get("name", getattr(tool, 'name', f"tool_{i}")),
                        "description": tool_metadata.get("description", getattr(tool, 'description', "工具描述未提供")),
                        "parameters": tool_metadata.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                    
                    # 验证并修正parameters格式，确保符合OpenAPI规范
                    if not isinstance(formatted_tool["parameters"], dict):
                        logger.warning(f"工具 {formatted_tool['name']} 的parameters不是字典类型，设置为默认值")
                        formatted_tool["parameters"] = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    
                    if "properties" not in formatted_tool["parameters"]:
                        logger.warning(f"工具 {formatted_tool['name']} 的parameters缺少properties字段，添加空properties")
                        formatted_tool["parameters"]["properties"] = {}
                        
                    if "type" not in formatted_tool["parameters"]:
                        logger.warning(f"工具 {formatted_tool['name']} 的parameters缺少type字段，设置为object")
                        formatted_tool["parameters"]["type"] = "object"
                    
                    tools_for_binding.append(formatted_tool)
                    logger.info(f"工具 {formatted_tool['name']} 成功格式化")
                except Exception as e:
                    logger.error(f"处理工具 {i+1} 时出错: {str(e)}")
                    logger.error(f"错误详情: {repr(e)}")
                    if hasattr(e, '__dict__'):
                        logger.error(f"错误属性: {e.__dict__}")
            
            # 如果没有有效的工具，使用空列表
            if not tools_for_binding:
                logger.warning("没有有效的工具可用于绑定，将使用空列表")
            else:
                logger.info(f"成功格式化 {len(tools_for_binding)} 个工具")
                
            # 绑定LLM和工具
            agent = (
                {
                    "input": RunnablePassthrough(),
                    "chat_history": lambda _: self.memory,
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
                }
                | prompt
                | llm.bind(functions=tools_for_binding)
                | OpenAIFunctionsAgentOutputParser()
            )
            
            # 创建Agent执行器
            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
            )
            
            logger.info("已成功设置Agent")
            return self.agent_executor
            
        except Exception as e:
            logger.error(f"设置Agent时出错: {str(e)}")
            logger.error(f"错误详情: {repr(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"错误属性: {e.__dict__}")
            raise
    
    async def process_query(self, query: str) -> str:
        """
        处理用户查询
        
        Args:
            query: 用户查询字符串
            
        Returns:
            处理结果
        """
        if not self.agent_executor:
            raise ValueError("请先设置Agent再处理查询")
            
        # 记录用户消息
        self.memory.append(HumanMessage(content=query))
        
        # 执行查询
        try:
            result = await self.agent_executor.ainvoke({"input": query})
            response = result["output"]
            
            # 记录AI回复
            self.memory.append(AIMessage(content=response))
            
            return response
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def direct_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        直接调用指定的工具（不通过Agent）
        
        Args:
            tool_name: 要调用的工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        if not self.tools:
            raise ValueError("请先加载工具再尝试直接调用")
        
        # 查找指定的工具
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            available_tools = ", ".join([t.name for t in self.tools])
            raise ValueError(f"找不到工具: {tool_name}。可用工具: {available_tools}")
        
        logger.info(f"直接调用工具: {tool_name}, 参数: {arguments}")
        
        try:
            # 调用工具
            result = await tool.ainvoke(arguments)
            logger.info(f"工具 {tool_name} 执行成功")
            return result
        except Exception as e:
            error_msg = f"调用工具 {tool_name} 时出错: {str(e)}"
            logger.error(error_msg)
            raise


async def main():
    """主函数"""
    # 通过环境变量读取配置文件路径
    config_path = os.getenv("MCP_CONFIG_PATH", "mcp-servers.json")
    
    # 初始化客户端
    client = LangChainMCPAPIExample(config_path=config_path)
    
    try:
        # 加载服务器配置
        await client.load_server_configs()
        
        server_name = os.getenv("MCP_SERVER_NAME", "common-api-server")
        if server_name not in client.server_configs:
            raise ValueError(f"找不到服务器配置: {server_name}")
            
        server_config = client.server_configs[server_name]
        logger.info(f"正在连接到MCP服务器: {server_name} ({server_config.get('url', 'URL未指定')})")
        
        # 使用上下文管理器自动处理连接和关闭
        async with MultiServerMCPClient({server_name: server_config}) as mcp_client:
            client.mcp_client = mcp_client
            logger.info(f"已成功连接到MCP服务器: {server_name}")
            
            # 加载MCP工具 - 注意get_tools()不是异步方法，不需要await
            client.tools = mcp_client.get_tools()
            logger.info(f"已从MCP服务器加载工具: {len(client.tools)}个工具")
            
            # 输出工具名称以供参考
            for i, tool in enumerate(client.tools):
                logger.info(f"工具 {i+1}: {tool.name} - {tool.description[:50]}...")
            
            # 检测是否使用DashScope API
            base_url = os.getenv("OPENAI_BASE_URL", "")
            is_dashscope = "dashscope" in base_url.lower() if base_url else False
            
            # 如果是DashScope API且存在工具，询问用户是否禁用工具
            enable_tools = True
            if is_dashscope and client.tools:
                logger.warning("检测到您使用的是阿里云DashScope API，该API可能与某些工具格式不兼容")
                response = input("是否禁用工具以使用纯聊天模式？(y/n): ").strip().lower()
                if response in ["y", "yes"]:
                    enable_tools = False
                    logger.info("已禁用工具，将使用纯聊天模式")
                else:
                    logger.info("将尝试使用工具模式，如遇到问题，请重启程序并选择禁用工具")
            
            # 设置Agent
            client.setup_agent(enable_tools=enable_tools)
            
            print("\n📝 MCP工具已加载完成，可以开始对话（输入'退出'结束）")
            
            # 处理用户输入
            while True:
                user_input = input("\n👤 请输入您的问题: ")
                if user_input.lower() in ["退出", "exit", "quit"]:
                    print("👋 再见！")
                    break
                    
                print("🤔 正在思考...")
                response = await client.process_query(user_input)
                print(f"\n🤖 {response}")
                
                # 示例：直接调用工具的代码（取消注释使用）
                # try:
                #     result = await client.direct_call_tool(
                #         "mcp_common_api_server_list_api_names",  # 替换为实际的工具名称
                #         {"random_string": ""}  # 替换为实际的参数
                #     )
                #     print(f"直接调用工具结果: {result}")
                # except Exception as e:
                #     print(f"直接调用工具失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"运行时错误: {str(e)}")
        # 上下文管理器会自动关闭连接，无需显式关闭


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 