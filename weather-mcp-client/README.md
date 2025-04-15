# 通用 MCP 客户端与天气服务器

基于 Model Context Protocol (MCP) 实现的通用客户端和天气服务器示例：

- **服务器**：使用高德地图天气API提供中国城市的实时天气信息和天气预报
- **客户端**：通用MCP客户端，可连接任何MCP服务器，并可选集成大模型API

## 功能特点

### 天气服务器

此MCP服务器提供两个工具：

* **get_weather**  
  * 获取中国城市的当前天气信息 
  * 输入参数: `city` (字符串): 城市名称，如"北京"、"上海"

* **get_weather_forecast**  
  * 获取中国城市的天气预报信息  
  * 输入参数: `city` (字符串): 城市名称，如"北京"、"上海"

### 通用MCP客户端

* 支持连接多个MCP服务器
* 自动发现并组织所有可用工具
* 通过命令行直接调用工具
* 通过OpenAI兼容接口调用大模型API（可选）
* 支持通过配置文件加载服务器配置

## 安装要求

* Python 3.10 或更高版本
* mcp[cli] >= 1.2.0
* httpx >= 0.27.0
* openai >= 1.5.0 (如需大模型集成)

## 安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 配置API密钥

### 高德地图API密钥（天气服务器）

1. 访问[高德开放平台](https://lbs.amap.com/)注册账号并申请Web服务API密钥
2. 在配置文件中添加密钥，或设置环境变量：

```bash
# Linux/macOS
export AMAP_API_KEY="你的API密钥"

# Windows
set AMAP_API_KEY=你的API密钥
```

### 通义千问API密钥（如需大模型集成）

1. 访问[阿里云百炼平台](https://dashscope.aliyun.com/)注册账号并申请API密钥
2. 设置API密钥环境变量：

```bash
# Linux/macOS
export DASHSCOPE_API_KEY="你的API密钥"

# Windows
set DASHSCOPE_API_KEY=你的API密钥
```

## 配置文件

客户端使用配置文件加载MCP服务器信息。默认配置文件名为`mcp-servers.json`，位于当前目录。配置文件格式如下：

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [
        "weather-mcp/weather.py"
      ],
      "env": {
        "AMAP_API_KEY": "你的高德地图API密钥"
      }
    },
    "another-server": {
      "command": "python",
      "args": [
        "path/to/another/server.py"
      ],
      "env": {
        "ENV_VAR1": "value1",
        "ENV_VAR2": "value2"
      }
    }
  }
}
```

配置参数说明：
- `command`: 启动服务器的命令
- `args`: 命令参数列表
- `env`: 环境变量字典

可以在配置文件中添加任意数量的MCP服务器。

## 使用方法

### 天气服务器单独运行

天气服务器通常不需要直接启动，它会被客户端根据配置文件自动启动。但如果需要单独测试，可以执行：

```bash
python weather.py
```

### 客户端

默认使用当前目录下的`mcp-servers.json`配置文件：

```bash
python client.py
```

也可以指定配置文件路径：

```bash
python client.py 路径/到/配置文件.json
```

#### 客户端命令

客户端启动后，可以使用以下命令格式：

1. **直接调用工具**：
   ```
   /<工具名> [参数]
   ```
   例如：`/get_weather city=北京`

2. **获取帮助**：
   ```
   /帮助 或 /help
   ```
   
3. **自然语言查询**（需要配置大模型）：
   直接输入问题，如：`北京今天的天气怎么样？`

4. **退出**：
   ```
   退出 或 exit 或 quit
   ```

## 客户端架构

客户端采用模块化设计，主要组件包括：

1. **服务器连接管理**：负责连接并管理多个MCP服务器
2. **工具发现与组织**：从所有服务器收集工具并统一管理
3. **命令解析器**：解析用户输入的命令和参数
4. **大模型集成**：将查询和上下文发送给大模型API

客户端启动流程：
1. 加载配置文件获取所有服务器配置
2. 连接到每个配置的MCP服务器
3. 收集所有服务器提供的工具
4. 处理用户输入：
   - 命令模式：直接调用相应工具
   - 自然语言模式：通过大模型处理

## 客户端实现细节

大模型集成使用OpenAI SDK通过兼容接口调用通义千问API：

```python
# 初始化OpenAI客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用通义千问API
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "用户问题"}
    ]
)
```

### 与Claude Desktop集成

要在Claude Desktop中使用天气服务器，需要编辑Claude Desktop的配置文件：

#### macOS/Linux
```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

#### Windows
```bash
code $env:AppData\Claude\claude_desktop_config.json
```

将以下JSON片段添加到配置文件中：

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": [
        "/完整/路径/到/weather-mcp/weather.py"
      ],
      "env": {
        "AMAP_API_KEY": "你的API密钥"
      }
    }
  }
}
```

请确保替换为你的实际路径和API密钥。

## 扩展开发

### 添加新的MCP服务器

1. 实现符合MCP协议的服务器
2. 在`mcp-servers.json`中添加服务器配置
3. 重启客户端即可使用新服务器提供的工具

### 集成其他大模型

要集成其他大模型，可以修改`MCPClient`类中的`query_llm`方法，添加对应的处理函数。

## 数据来源

* 天气数据：高德地图天气API
* 智能对话：阿里云通义千问API (通过OpenAI兼容接口)

# 天气MCP客户端

这个项目包含了两种不同的MCP（Model Context Protocol）客户端实现方式，用于连接MCP服务器并访问其工具。

## 项目结构

- `langchain_client.py` - 基于原始MCP库的客户端实现
- `langchain_mcp_example.py` - 使用langchain-mcp-adapters的简化客户端实现
- `requirements.txt` - 项目依赖项
- `mcp-servers.json` - MCP服务器配置文件

## 依赖项安装

```bash
pip install -r requirements.txt
```

## 服务器配置

编辑`mcp-servers.json`文件，配置你的MCP服务器:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["../weather.py"],
      "env": {
        "WEATHER_API_KEY": "your_api_key"
      }
    },
    "math": {
      "command": "python",
      "args": ["../math_server.py"]
    }
  }
}
```

## 使用方法

### 1. 设置环境变量

大模型API密钥可以通过环境变量设置:

```bash
# Windows
$env:OPENAI_API_KEY="your_api_key"

# Linux/Mac
export OPENAI_API_KEY="your_api_key"
```

或者创建`.env`文件:

```
OPENAI_API_KEY=your_api_key
```

### 2. 使用原始客户端

```bash
python langchain_client.py
```

原始客户端提供以下功能:
- 支持多个MCP服务器连接
- 命令行工具调用格式: `/工具名 参数`
- 自然语言查询处理
- 多轮对话支持

### 3. 使用langchain-mcp-adapters客户端

```bash
python langchain_mcp_example.py
```

这个简化客户端使用langchain-mcp-adapters库，提供以下功能:
- 更简洁的代码
- 原生LangChain集成
- 自动工具加载
- ReAct代理模式

## 两种实现的对比

| 特性 | 原始客户端 | langchain-mcp-adapters客户端 |
|-----|----------|---------------------------|
| 代码复杂度 | 较高 | 较低 |
| 自定义能力 | 强 | 中等 |
| 工具调用方式 | 命令行格式和自然语言 | 主要是自然语言 |
| 异常处理 | 详细 | 基本 |
| 适合场景 | 需要深度定制的项目 | 快速原型开发 |

## 关于MCP

Model Context Protocol (MCP) 是由Anthropic开发的开源协议，用于连接大型语言模型与外部工具和数据源。它提供了标准化的接口，使LLM能够安全地访问和使用各种外部功能。 