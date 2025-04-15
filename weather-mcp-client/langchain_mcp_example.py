#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangChain MCP适配器示例脚本

这个脚本展示了如何使用langchain-mcp-adapters库连接MCP服务器，
加载工具，并使用LangChain Agent处理用户查询。
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

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
from langchain_mcp_adapters import connect_to_mcp_server, load_mcp_tools
from langchain_mcp_adapters.tools import MCPToolSpec, convert_mcp_tool_to_langchain

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LangChainMCPExample:
    """
    使用LangChain适配器处理MCP工具的示例类
    
    这个类展示了如何:
    1. 连接到MCP服务器
    2. 加载MCP工具
    3. 将MCP工具转换为LangChain工具
    4. 创建LangChain Agent
    5. 处理用户查询
    """
    
    def __init__(self, config_path: str):
        """
        初始化LangChain MCP示例
        
        Args:
            config_path: 配置文件路径
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
                self.server_configs = json.load(f)
                logger.info(f"已加载MCP服务器配置: {self.config_path}")
                return self.server_configs
        except Exception as e:
            logger.error(f"加载服务器配置时发生错误: {str(e)}")
            raise
    
    async def connect_and_load_tools(self, server_name: str) -> List[BaseTool]:
        """
        连接到MCP服务器并加载工具
        
        Args:
            server_name: 要连接的服务器名称
            
        Returns:
            转换后的LangChain工具列表
        """
        if not self.server_configs:
            await self.load_server_configs()
            
        if server_name not in self.server_configs:
            raise ValueError(f"找不到服务器配置: {server_name}")
            
        server_config = self.server_configs[server_name]
        logger.info(f"正在连接到MCP服务器: {server_name}")
        
        try:
            # 使用langchain-mcp-adapters连接到MCP服务器
            self.mcp_client = await connect_to_mcp_server(server_config)
            logger.info(f"已成功连接到MCP服务器: {server_name}")
            
            # 加载MCP工具
            mcp_tools = await load_mcp_tools(self.mcp_client)
            logger.info(f"已从MCP服务器加载工具: {len(mcp_tools)}个工具")
            
            # 将MCP工具转换为LangChain工具
            self.tools = [convert_mcp_tool_to_langchain(tool) for tool in mcp_tools]
            logger.info(f"已转换MCP工具为LangChain工具: {len(self.tools)}个工具")
            
            return self.tools
        except Exception as e:
            logger.error(f"连接MCP服务器或加载工具时出错: {str(e)}")
            raise
    
    def setup_agent(self, model_name: str = "gpt-4o", temperature: float = 0) -> AgentExecutor:
        """
        设置LangChain Agent
        
        Args:
            model_name: 要使用的模型名称
            temperature: 模型温度参数
            
        Returns:
            配置好的AgentExecutor
        """
        if not self.tools:
            raise ValueError("请先加载工具再设置Agent")
            
        # 创建LLM
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，能够利用各种工具帮助用户解决问题。根据用户的需求，选择合适的工具并执行。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 绑定LLM和工具
        agent = (
            {
                "input": RunnablePassthrough(),
                "chat_history": lambda _: self.memory,
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm.bind(functions=[tool.metadata for tool in self.tools])
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
            result = await self.agent_executor.ainvoke({"input": query, "intermediate_steps": []})
            response = result["output"]
            
            # 记录AI回复
            self.memory.append(AIMessage(content=response))
            
            return response
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg


async def main():
    """主函数"""
    # 初始化客户端
    client = LangChainMCPExample("config.json")
    
    try:
        # 连接到MCP服务器并加载工具
        await client.connect_and_load_tools("weather")
        
        # 设置Agent
        client.setup_agent()
        
        print("📝 MCP工具已加载完成，可以开始对话（输入'退出'结束）")
        
        # 处理用户输入
        while True:
            user_input = input("👤 请输入您的问题: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("👋 再见！")
                break
                
            response = await client.process_query(user_input)
            print(f"🤖 {response}")
    except Exception as e:
        logger.error(f"运行时错误: {str(e)}")
    finally:
        # 这里可以添加清理代码
        if client.mcp_client:
            await client.mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main()) 