# LangChain / LangGraph 学习 Demo

用**本机现有环境**写的 8 个可运行 demo，**不需要再装任何包**。
每个 demo 都支持 `--fake` 离线模式，不消耗任何额度就能把流程跑通。

---

## 1. 环境

已装并直接可用：

| 包 | 版本 | 用途 |
|---|---|---|
| langchain | 1.4.0 | 链式编排、Agent |
| langchain-core | 1.6.2 | Runnable、Prompt、Parser、Document、Retriever 基类 |
| langchain-openai | 1.6.2 | ChatOpenAI（也用于 MiniMax 等 OpenAI 兼容端点） |
| langchain-deepseek | 1.1.0 | ChatDeepSeek |
| langgraph | 1.2.11 | StateGraph、checkpoint、interrupt |
| langgraph-checkpoint | 4.2.0 | InMemorySaver |
| langgraph-prebuilt | 1.1.0 | 内置节点 |
| langsmith | 0.12.4 | 追踪（可选） |
| python-dotenv | 已装 | 读 .env |

**没装**（所以本系列里对应的能力都是手写替代品，demo 里都标注了真实项目该用什么）：

`langchain-text-splitters`、`langchain-community`、`langchain-classic`、
所有向量库（chromadb / faiss）、`langgraph-checkpoint-sqlite`。

---

## 2. 怎么跑

```powershell
cd PythonProject\LangChain\demos

# 离线模式：假模型，流程照跑，不花额度
python 01_llm_basics.py --fake

# 真实模式：需要先配好密钥
Copy-Item .env.example .env
notepad .env          # 填入 DEEPSEEK_API_KEY（或 MINIMAX_API_KEY）
python 01_llm_basics.py
```

密钥只从 `.env` 读，`.env` 已在仓库根目录 `.gitignore` 第 108 行被忽略，不会进 git。

---

## 3. 学习路线

按顺序看，每个 demo 都能独立运行。

| # | 文件 | 主题 | 关键 API |
|---|---|---|---|
| 01 | `01_llm_basics.py` | 大模型基础调用 | `invoke` / `stream` / `batch`、`bind()`、token 用量 |
| 02 | `02_prompt_lcel.py` | 提示词与链式编排 | `ChatPromptTemplate`、`\|` 管道、`RunnablePassthrough` / `Lambda` / `Parallel` / `Branch` |
| 03 | `03_structured_output.py` | 结构化输出 | `with_structured_output`、`PydanticOutputParser`、`JsonOutputParser` |
| 04 | `04_tools.py` | 工具调用 | `@tool` 三种写法、`bind_tools`、`tool_calls`、手工工具循环 |
| 05 | `05_agent.py` | Agent | `create_agent`、`system_prompt`、`checkpointer` + `thread_id`、`middleware` |
| 06 | `06_langgraph_basics.py` | 图编排 | `StateGraph`、`add_conditional_edges`、循环、`recursion_limit`、并行分支、reducer |
| 07 | `07_langgraph_memory_hitl.py` | 记忆与人工介入 | `InMemorySaver`、`get_state_history`、time travel、`interrupt` + `Command(resume=...)` |
| 08 | `08_rag_handmade.py` | 手写 RAG | 手写切分器、`Document`、`BaseRetriever` 子类 + BM25、检索拼上下文带引用 |

配套的 `common.py` 提供：`.env` 加载、模型工厂（DeepSeek / MiniMax / OpenAI 兼容）、
`ScriptedChatModel`（脚本化假模型，用来离线演示 `tool_calls`）。

---

## 4. 这套代码踩过的坑（都是本机实测出来的）

1. **`langchain.prompts` 在 1.x 没有了** —— 必须 `from langchain_core.prompts import ChatPromptTemplate`。
   同理 `langchain.chains` 也没了，老教程的 `LLMChain` 需要装 `langchain-classic`。
2. **`model.bind(system=...)` 无效** —— `ChatOpenAI` / `ChatDeepSeek` 都没有 `system` 字段，
   bind 进去会被当成 API 扩展参数发出去。要固定人格请用 `SystemMessage` 或 `ChatPromptTemplate`。
3. **`RunnableBranch(*branches)` 的最后一个参数不是二元组** —— 前面的才是 `(条件, Runnable)`，
   最后一个直接是默认 Runnable。
4. **Windows 控制台是 GBK 编码** —— 打印 `✓` 这类符号会直接抛 `UnicodeEncodeError` 让脚本崩掉。
   `common.py` 里已经把 stdout 降级成 `errors="replace"`，并且全部输出字符都保证 GBK 可编码。
5. **LangGraph 默认 `recursion_limit` 是 10000+** —— 自己写的循环一定要有 termination 条件，
   否则会一直转到内存爆掉。
6. **`with_structured_output` 走 tool calling** —— 离线假模型不支持，会抛 `NotImplementedError`，
   所以 demo 03 在 `--fake` 下会自动跳到 `PydanticOutputParser` 那条路。

---

## 5. 和仓库里原有 4 个脚本的关系

`PythonProject/LangChain/` 下原有的 `test1.py` / `test2.py` / `test3.py` / `tool1.py`
写法本身没问题（都是 1.x 新 API），只是缺配置和上下文。本目录的 demo 等于是把那几个脚本
扩成了完整、可复用的版本：

| 原脚本 | 对应的 demo |
|---|---|
| `test1.py` 直调模型 + 输出解析 | `01_llm_basics.py` |
| `test2.py` 链式执行 | `02_prompt_lcel.py` |
| `test3.py` `init_chat_model` 可配置模型 | `01` 里的 `bind()` 部分 |
| `tool1.py` `@tool` 三种写法 | `04_tools.py` |

---

## 6. 下一步（等你要做项目时再看）

- **要持久化对话**：装 `langgraph-checkpoint-sqlite`，把 `InMemorySaver()` 换掉即可。
- **要做 RAG**：装 `langchain-text-splitters` + `langchain-community` + 一个向量库，
  demo 08 里那张对照表就是替换清单。
- **要跟老教程**：`pip install langchain-classic`。
- **要装包**：在**普通 PowerShell 窗口**里执行，不要在 DSH 会话里执行
  （DSH 沙箱里 pip 解包 wheel 会报 Permission denied）。
