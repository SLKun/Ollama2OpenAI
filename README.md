# Ollama2OpenAI

一个将 Ollama 格式的 API 请求转发到 OpenAI 兼容接口的服务。

## 功能特点 ✨

- 支持将OpenAI格式请求转化到VSCode Github Copilot的Ollama上游
- 支持LobeHub的Ollama上游
- 支持CherryStudio的Ollama上游
- 提供一个简单的WEB页面进行配置

目前已兼容的API接口:

- `/api/tags`
- `/api/show`
- `/api/chat`
- `/api/generate`
- `/api/embedding`
- `/v1/models`
- `/v1/chat/completions`

## 界面预览 ✨

![image](https://github.com/user-attachments/assets/e58293d0-c2ac-442f-be5c-48a0c6de4220)


## 快速开始 🚀

### 本地构建（开发者）

```bash
# 构建镜像
docker build -t ollama2openai .

# 创建数据目录
mkdir -p data

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  ollama2openai
```

配置文件会保存在 `data` 目录下，重启容器时会自动加载。

### 手动安装

1. 克隆仓库：
```bash
git clone https://github.com/slkun/ollama2openai.git
cd ollama2openai
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行服务：
```bash
python main.py
```

## 配置说明 ⚙️

访问 `http://localhost:8000/login` 进入配置界面，可配置以下内容：

- 管理密码：用于登录配置界面, 默认密码为`admin`
- OpenAI API Key：用于访问 OpenAI 兼容接口
- Ollama API Key：用于 Ollama API 认证（可选）
- OpenAI API Base URL：OpenAI 兼容接口的基础 URL
- 模型映射：配置 Ollama 模型名称到 OpenAI 模型的映射关系

### 模型映射

你可以为 OpenAI 的模型配置在 Ollama 中显示的别名。例如：

```json
{
  "llama2": "gpt-4",
  "mistral": "gpt-3.5-turbo"
}
```

配置界面支持：
- 点击可用模型列表自动创建映射
- 自动生成规范的 Ollama 别名
- 直观的映射关系管理

### 键盘快捷键

- `Alt + 1`: 聚焦管理密码
- `Alt + 2`: 聚焦 OpenAI API Key
- `Alt + 3`: 聚焦 Ollama API Key
- `Alt + 4`: 聚焦 Base URL
- `Alt + 5`: 添加新映射
- `Alt + S`: 保存配置
- `Alt + T`: 切换主题
- `Alt + H`: 显示/隐藏快捷键面板
