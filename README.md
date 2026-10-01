# 文献 Agent (Literature Agent)

一个基于 RAG（检索增强生成）的文献智能助手，支持 PDF 文献的索引、检索和对话问答。

## 功能特性

- **PDF 文献解析** — 使用 `pdfplumber` 和 `pypdf` 提取文献内容
- **向量检索** — 基于 ChromaDB + Sentence Transformers 构建语义搜索
- **对话问答** — 通过 LangGraph 编排 Agent，支持多轮对话
- **Web 界面** — 内置 FastAPI Web 应用，提供交互式对话体验
- **会话持久化** — SQLite 存储对话历史
- **评估工具** — 内置评测脚本，方便测试检索质量

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.13+ |
| 框架 | LangChain、LangGraph、FastAPI |
| 向量库 | ChromaDB |
| 嵌入模型 | Sentence Transformers |
| 文档解析 | pdfplumber、pypdf |
| 包管理 | uv |

## 项目结构

```
my-agent-program/
├── src/my_agent_program/   # 主包入口
├── papers/                 # 文献 PDF 存放目录
├── chroma_db/              # ChromaDB 向量索引
├── templates/              # Web 应用模板
├── tests/                  # 测试文件
├── web_app.py              # FastAPI Web 应用
├── indexer.py              # 文献索引构建脚本
├── retriever.py            # 检索器
├── build_index.py          # 索引构建入口
├── run_eval.py             # 评估脚本
├── conversation.db         # 对话历史数据库
└── pyproject.toml          # 项目配置
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example`（如有）或手动创建 `.env`：

```env
# 使用本地模型时设置（可选）
# HF_HUB_OFFLINE=1

# 使用 OpenAI API 时设置
OPENAI_API_KEY=your-api-key
```

### 3. 构建文献索引

将 PDF 文献放入 `papers/` 目录，然后运行：

```bash
uv run python build_index.py
```

### 4. 启动 Web 应用

```bash
uv run python web_app.py
```

访问 `http://localhost:8000` 即可使用。

### 5. 运行测试

```bash
uv run pytest
```

## 使用提示

- **添加文献**：将 PDF 放入 `papers/` 目录后重新运行 `build_index.py` 即可更新索引。
- **对话历史**：所有对话记录保存在 `conversation.db` 中，支持多轮上下文。

## 关于 `HF_HUB_OFFLINE`

项目中多处设置了 `os.environ["HF_HUB_OFFLINE"] = "1"`，涉及以下文件：

| 文件 | 位置 |
|------|------|
| `indexer.py` | 第 8 行 |
| `retriever.py` | 第 2 行 |
| `langgraph_demo.py` | 第 27 行 |

### 作用

`HF_HUB_OFFLINE = "1"` 会让 Hugging Face 库**完全使用本地缓存的模型**，不尝试联网下载。项目使用的嵌入模型是 `all-MiniLM-L6-v2`。

### 使用场景

| 场景 | 建议 |
|------|------|
| 模型已下载到本地 | 保持 `= "1"`，避免每次启动都检查更新 |
| 首次使用 / 模型未下载 | **注释掉或删除**该行，让程序自动从 Hugging Face 下载模型 |
| 网络不稳定 | 先联网下载模型，再设置为 `"1"` 离线使用 |

### 如何切换

```python
# 离线模式（使用本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"

# 在线模式（允许下载）— 注释掉上面那行即可
# os.environ["HF_HUB_OFFLINE"] = "1"
```

> **注意**：如果设置为 `"1"` 但本地没有缓存模型，程序会报错。首次使用请先确保模型已下载。

## License

MIT
