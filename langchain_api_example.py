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
from functools import wraps

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
            
            在使用工具时，请注意：
            1. 确保严格按照工具要求的格式提供参数
            2. 工具返回的内容可能是复杂的JSON，需要你解析和理解
            3. 如果工具调用失败，请分析错误信息，修正参数后重试
            4. 返回给用户的内容应当简洁易懂，不要返回原始的JSON
            
            遇到错误时，请进行如下处理：
            - 如果参数错误，检查参数类型和必填项，然后重试
            - 如果API端点返回错误，解释错误原因并提供替代解决方案
            - 如果你不确定如何使用某个工具，可以用简单的示例尝试，或者选择其他更熟悉的工具
            
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
            # 添加一个安全的工具包装器类，适应各种工具类型
            class SafeToolWrapper:
                """
                安全的工具包装器，处理各种工具类型和异常情况
                """
                
                @staticmethod
                def wrap_tools(tools):
                    """包装工具列表，返回包装后的工具列表"""
                    wrapped_tools = []
                    
                    for tool in tools:
                        try:
                            # 确定工具类型和调用方法
                            tool_name = getattr(tool, 'name', str(tool))
                            tool_type = type(tool).__name__
                            logger.info(f"包装工具: {tool_name}, 类型: {tool_type}")
                            
                            # 根据工具类型选择适当的包装方法
                            wrapped_tool = SafeToolWrapper._wrap_tool(tool)
                            wrapped_tools.append(wrapped_tool)
                            logger.info(f"工具 {tool_name} 包装成功")
                        except Exception as e:
                            logger.error(f"包装工具 {getattr(tool, 'name', str(tool))} 失败: {str(e)}")
                            # 如果包装失败，添加原始工具
                            wrapped_tools.append(tool)
                    
                    return wrapped_tools
                
                @staticmethod
                def _wrap_tool(tool):
                    """包装单个工具"""
                    # 为各种可能的执行方法添加异常处理和响应处理
                    
                    # 包装 _run 方法
                    if hasattr(tool, "_run"):
                        original_run = tool._run
                        
                        @wraps(original_run)
                        def safe_run(*args, **kwargs):
                            try:
                                result = original_run(*args, **kwargs)
                                # 对结果进行预处理，确保格式正确
                                return SafeToolWrapper._process_response(result, tool.name)
                            except Exception as e:
                                logger.error(f"工具 {tool.name} 执行出错: {str(e)}")
                                error_msg = f"工具执行错误 ({tool.name}): {str(e)}"
                                # 返回元组格式
                                return (error_msg, str(e))
                        
                        tool._run = safe_run
                    
                    # 包装 _arun 方法
                    if hasattr(tool, "_arun"):
                        original_arun = tool._arun
                        
                        @wraps(original_arun)
                        async def safe_arun(*args, **kwargs):
                            try:
                                result = await original_arun(*args, **kwargs)
                                # 对结果进行预处理，确保格式正确
                                return SafeToolWrapper._process_response(result, tool.name)
                            except Exception as e:
                                logger.error(f"工具 {tool.name} 异步执行出错: {str(e)}")
                                error_msg = f"工具异步执行错误 ({tool.name}): {str(e)}"
                                # 返回元组格式
                                return (error_msg, str(e))
                        
                        tool._arun = safe_arun
                    
                    return tool
                
                @staticmethod
                def _process_response(response, tool_name):
                    """处理工具响应，确保格式正确且易于解析"""
                    try:
                        # 如果响应已经是元组格式，检查是否符合(content, artifact)格式
                        if isinstance(response, tuple) and len(response) == 2:
                            # 已经是正确格式，直接返回
                            return response
                            
                        processed_content = None
                        original_response = response  # 保存原始响应用于artifact
                        
                        # 如果是JSON字符串，尝试解析再重新序列化，以确保格式一致
                        if isinstance(response, str):
                            try:
                                if response.startswith('[') and response.endswith(']'):
                                    # 处理JSON数组
                                    json_array = json.loads(response)
                                    # 对数组中每个对象进行处理，特别处理中文和特殊字符
                                    simplified_results = []
                                    for item in json_array:
                                        if isinstance(item, str):
                                            try:
                                                # 尝试解析嵌套的JSON字符串
                                                json_item = json.loads(item)
                                                simplified_results.append(json_item)
                                            except:
                                                simplified_results.append(item)
                                        else:
                                            simplified_results.append(item)
                                    
                                    # 将复杂对象转换为简洁格式
                                    simplified_text = ""
                                    for item in simplified_results:
                                        if isinstance(item, dict) and "api_name" in item and "description" in item:
                                            desc = item.get("description", "")
                                            # 移除不必要的转义字符
                                            if isinstance(desc, str):
                                                desc = desc.encode().decode('unicode_escape')
                                            simplified_text += f"- {item['api_name']}: {desc}\n"
                                        else:
                                            simplified_text += f"- {str(item)}\n"
                                    
                                    processed_content = simplified_text
                                elif response.startswith('{') and response.endswith('}'):
                                    # 处理单个JSON对象
                                    json_obj = json.loads(response)
                                    
                                    # 如果是API文档，进行特殊处理
                                    if "api_name" in json_obj and "description" in json_obj and "doc" in json_obj:
                                        api_name = json_obj.get("api_name", "")
                                        desc = json_obj.get("description", "")
                                        doc = json_obj.get("doc", "")
                                        
                                        # 移除不必要的转义字符
                                        if isinstance(desc, str):
                                            desc = desc.encode().decode('unicode_escape')
                                        if isinstance(doc, str):
                                            doc = doc.encode().decode('unicode_escape')
                                        
                                        processed_content = f"API名称: {api_name}\n描述: {desc}\n\n文档:\n{doc}"
                                    else:
                                        # 将JSON对象转为字符串，确保格式正确
                                        processed_content = json.dumps(json_obj, ensure_ascii=False, indent=2)
                            except Exception as e:
                                logger.warning(f"处理工具 {tool_name} 的JSON响应时出错: {str(e)}")
                                # 如果JSON解析失败，使用原始响应
                                processed_content = response
                        
                        # 对于其他类型的响应，转换为字符串
                        if processed_content is None:
                            if not isinstance(response, str):
                                response_str = str(response)
                                # 如果是复杂对象的字符串表示，尝试美化
                                if response_str.startswith('{') or response_str.startswith('['):
                                    try:
                                        processed_content = json.dumps(response, ensure_ascii=False, indent=2)
                                    except:
                                        processed_content = response_str
                                else:
                                    processed_content = response_str
                            else:
                                processed_content = response
                        
                        # 返回符合response_format='content_and_artifact'格式的元组
                        return (processed_content, original_response)
                    except Exception as e:
                        logger.error(f"处理工具 {tool_name} 响应时发生未知错误: {str(e)}")
                        error_msg = f"处理响应出错，原始响应: {str(response)[:100]}..."
                        # 即使出错也返回元组格式
                        return (error_msg, response)
            
            # 使用新的包装器包装工具
            try:
                logger.info(f"开始包装 {len(self.tools)} 个工具...")
                self.tools = SafeToolWrapper.wrap_tools(self.tools)
                logger.info("所有工具包装完成")
            except Exception as e:
                logger.error(f"工具包装过程中出错: {str(e)}")
                logger.error(f"将继续使用原始工具")
            
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
                    
                    # 深度验证并修正parameters格式，确保符合OpenAPI规范
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
                    
                    # 确保properties是一个对象而不是数组或其他类型
                    if not isinstance(formatted_tool["parameters"]["properties"], dict):
                        logger.warning(f"工具 {formatted_tool['name']} 的properties不是字典类型，设置为空字典")
                        formatted_tool["parameters"]["properties"] = {}
                    
                    # 确保required是一个数组
                    if "required" in formatted_tool["parameters"] and not isinstance(formatted_tool["parameters"]["required"], list):
                        logger.warning(f"工具 {formatted_tool['name']} 的required不是数组类型，设置为空数组")
                        formatted_tool["parameters"]["required"] = []
                    
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
            try:
                self.agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=self.tools,
                    verbose=True,
                    handle_parsing_errors=True,  # 启用解析错误处理，允许将错误传回模型
                    max_iterations=5,  # 允许最多重试5次
                    early_stopping_method="force",  # 使用force替代generate作为早期停止方法
                    return_intermediate_steps=True,  # 返回中间步骤以便分析
                )
                
                logger.info("已成功设置Agent")
                return self.agent_executor
            except Exception as e:
                logger.error(f"创建AgentExecutor时出错: {str(e)}")
                logger.error(f"错误详情: {repr(e)}")
                
                # 尝试使用更简单的配置创建Agent
                logger.info("尝试使用简化配置创建Agent...")
                
                try:
                    from langchain.agents import AgentType, initialize_agent
                    
                    self.agent_executor = initialize_agent(
                        tools=self.tools,
                        llm=llm,
                        agent=AgentType.OPENAI_FUNCTIONS,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=5,
                        early_stopping_method="force"  # 使用force替代generate
                    )
                    
                    logger.info("已使用简化配置成功设置Agent")
                    return self.agent_executor
                except Exception as fallback_error:
                    logger.error(f"使用简化配置创建Agent时也失败: {str(fallback_error)}")
                    raise e  # 重新抛出原始错误
            
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
            
            # 分析中间步骤，检查是否有工具调用错误
            if "intermediate_steps" in result:
                has_errors = False
                error_messages = []
                
                for step in result["intermediate_steps"]:
                    # 检查工具调用结果是否包含错误信息
                    if len(step) >= 2:
                        tool_result = step[1]
                        
                        # 如果工具结果是元组格式 (content, artifact)，提取content部分
                        if isinstance(tool_result, tuple) and len(tool_result) == 2:
                            tool_content = tool_result[0]
                        else:
                            tool_content = tool_result
                            
                        # 检测不同类型的错误信息
                        if isinstance(tool_content, str) and any(err in tool_content.lower() for err in ["错误", "error", "exception", "failed", "失败"]):
                            has_errors = True
                            error_message = f"工具 '{step[0].tool}' 执行出错: {tool_content}"
                            error_messages.append(error_message)
                            logger.warning(error_message)
                
                # 如果存在错误，添加到响应中
                if has_errors:
                    error_summary = "\n".join(error_messages)
                    logger.error(f"工具调用过程中发生错误: \n{error_summary}")
                    
                    # 将错误信息附加到响应中，使用明显的格式
                    if "错误" not in response and "error" not in response.lower():
                        response += f"\n\n⚠️ 系统提示: 执行过程中遇到以下问题:\n{error_summary}"
            
            # 记录AI回复
            self.memory.append(AIMessage(content=response))
            
            logger.info(f"查询处理完成，结果: {response[:100]}...")
            return response
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            logger.error(error_msg)
            
            # 记录详细的错误信息
            logger.error(f"错误详情: {repr(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"错误属性: {e.__dict__}")
                
            # 将错误信息添加到对话记忆中，让AI理解发生了什么
            error_content = f"抱歉，我在处理您的请求时遇到了技术问题。错误信息: {str(e)}"
            self.memory.append(AIMessage(content=error_content))
            
            # 返回友好的错误信息给用户
            return f"抱歉，处理您的请求时出现了问题: {str(e)}\n请尝试重新表述您的问题，或者尝试其他操作。"
    
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
            # 根据工具类型选择合适的调用方式
            result = None
            if hasattr(tool, 'ainvoke'):
                # 优先使用ainvoke方法
                result = await tool.ainvoke(arguments)
            elif hasattr(tool, '_arun'):
                # 使用_arun方法
                result = await tool._arun(**arguments)
            elif hasattr(tool, '_run'):
                # 尝试调用同步_run方法
                result = tool._run(**arguments)
            else:
                # 尝试使用__call__方法
                result = await tool.__call__(**arguments)
            
            logger.info(f"工具 {tool_name} 执行成功")
            
            # 处理元组格式的结果
            if isinstance(result, tuple) and len(result) == 2:
                # 只返回内容部分，忽略原始响应
                return result[0]
            
            return result
        except Exception as e:
            error_msg = f"调用工具 {tool_name} 时出错: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误详情: {repr(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"错误属性: {e.__dict__}")
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