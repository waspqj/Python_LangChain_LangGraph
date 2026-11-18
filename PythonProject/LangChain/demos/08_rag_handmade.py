"""demo 08 · 不装任何库，手写一个 RAG

本机没装 langchain-text-splitters / langchain-community / 任何向量库，
所以这一份**手写**切分器和检索器，把 RAG 的骨架跑通。

覆盖的知识点：
    1. Document 对象（langchain_core 自带）
    2. 手写切分器：段落 -> 句子 -> 定长滑窗（带重叠）
    3. 手写 BM25 检索，并且**包装成 BaseRetriever**（和真实检索器接口一致）
    4. RAG 链：检索 -> 拼上下文 -> 提示词 -> 生成 -> 带引用编号
    5. 真实项目里，这三步分别换成什么

运行：
    python 08_rag_handmade.py            # 用真实模型
    python 08_rag_handmade.py --fake     # 离线跑通检索与拼装
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from common import banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 知识库（真实项目里换成读文件 / 读数据库）
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: dict[str, str] = {
    "LangGraph 是什么": (
        "LangGraph 是一个用于构建有状态、多智能体应用的开源框架。"
        "它把应用建模成一张图：节点是执行单元，边决定流转顺序，"
        "状态在节点之间传递并被持久化。"
        "LangGraph 建立在 Pregel 的图执行模型之上，支持循环、条件分支、并行分支和中断恢复。"
    ),
    "LangChain 是什么": (
        "LangChain 是一个用于开发大模型应用的框架，核心理念是链式编排。"
        "它把模型调用、提示词模板、输出解析、工具调用、检索等能力统一成 Runnable 接口，"
        "让每个组件都能用竖线管道自由组合。"
    ),
    "Runnable 是什么": (
        "Runnable 是 LangChain 的统一执行接口，链、提示词、模型、输出解析器都是 Runnable。"
        "Runnable 之间可以用竖线管道组合，每个 Runnable 都支持 invoke、stream、batch 三种调用方式，"
        "因此同一条链既能单条调用也能批量并发。"
    ),
    "什么是 Checkpointer": (
        "Checkpointer 是 LangGraph 的状态持久化组件，会在每个超级步之后保存一份状态快照。"
        "配合 thread_id 可以把一份存档区分成多条独立会话。"
        "常用实现有内存版 InMemorySaver、SQLite 版 SqliteSaver 和 Postgres 版 PostgresSaver。"
    ),
    "什么是 Human in the loop": (
        "Human in the loop 指在流程中间插入人工判断。"
        "LangGraph 通过 interrupt 函数暂停图执行，等外部用 Command resume 送回一个值之后继续往下跑，"
        "适合需要审批的高风险操作。"
    ),
}


# ---------------------------------------------------------------------------
# 1. 手写切分器
# ---------------------------------------------------------------------------
def split_text(text: str, chunk_size: int = 80, chunk_overlap: int = 20) -> list[str]:
    """
    简化版的递归字符切分：

        段落 -> 句子 -> 定长滑窗（相邻块保留 chunk_overlap 个字符的重叠）

    重叠的作用：避免一句话正好被切断在两块之间，导致语义丢失。
    """
    chunks: list[str] = []

    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue

        # 按句号切
        sentences = [s for s in re.split(r"(?<=[。！？!?])", paragraph) if s.strip()]

        for sentence in sentences:
            if len(sentence) <= chunk_size:
                chunks.append(sentence.strip())
                continue

            # 还是太长 -> 定长滑窗
            step = max(chunk_size - chunk_overlap, 1)
            start = 0
            while start < len(sentence):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(sentence):
                    break
                start += step

    return chunks


def build_documents() -> list[Document]:
    """把知识库切成 Document 列表，metadata 里放来源信息。"""
    docs: list[Document] = []
    for title, text in KNOWLEDGE_BASE.items():
        for i, chunk in enumerate(split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": title, "chunk_id": i},
                )
            )
    return docs


# ---------------------------------------------------------------------------
# 2. 手写 BM25 检索，并包装成 BaseRetriever
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文按字切，英文数字按词切。"""
    return TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """
    继承 BaseRetriever，接口和真实检索器（Chroma / FAISS）完全一致：
        retriever.invoke("问题") -> list[Document]

    换成真实向量库时，上层代码一行都不用改。
    """

    docs: list[Document]
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        tokenized = [tokenize(d.page_content) for d in self.docs]
        n = len(tokenized)
        avgdl = sum(len(d) for d in tokenized) / max(n, 1)
        df = Counter()
        for doc in tokenized:
            df.update(set(doc))

        query_tokens = tokenize(query)
        scored: list[tuple[float, int]] = []

        for i, doc_tokens in enumerate(tokenized):
            tf = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl)
                score += idf * freq * 2.5 / denom
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        return [self.docs[i] for _, i in scored[: self.k]]


# ---------------------------------------------------------------------------
# 3. 组装 RAG 链
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个只根据给定资料回答问题的助手。\n"
            "资料中每段前面有 [编号]，引用时请写成 [编号] 的形式。\n"
            "如果资料里没有答案，就直接说资料里没有，不要编。\n\n"
            "资料：\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_context(docs: list[Document]) -> str:
    """把检索结果拼成带编号的上下文字符串。"""
    return "\n\n".join(
        f"[{i + 1}] （来源：{d.metadata.get('source', '未知')}）{d.page_content}"
        for i, d in enumerate(docs)
    )


def demo_retrieval(docs: list[Document]) -> None:
    hr("1. 切分与检索")

    print(f"  知识库条目数 : {len(KNOWLEDGE_BASE)}")
    print(f"  切分后块数   : {len(docs)}")
    print("  前 3 块示例:")
    for d in docs[:3]:
        print(f"    - ({d.metadata['source']}#{d.metadata['chunk_id']}) {d.page_content[:40]}...")

    retriever = BM25Retriever(docs=docs, k=2)
    hits = retriever.invoke("LangGraph 怎么保存状态")
    print()
    print("  查询 'LangGraph 怎么保存状态' 命中：")
    for i, d in enumerate(hits):
        print(f"    [{i + 1}] ({d.metadata['source']}) {d.page_content[:50]}...")


def demo_rag_chain(model, docs: list[Document]) -> None:
    hr("2. RAG 链")

    retriever = BM25Retriever(docs=docs, k=3)

    # RunnablePassthrough.assign 先把 context 塞进输入，再交给提示词
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_context(retriever.invoke(x["question"]))
            )
        )
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )

    question = "LangGraph 的 checkpointer 是干什么的？"
    print(f"  问题 : {question}")
    print(f"  回答 : {chain.invoke({'question': question})}")

    print()
    print("  如果还想拿到引用来源，把链拆成两段即可：")
    retrieved = retriever.invoke(question)
    prompt_value = RAG_PROMPT.invoke(
        {"question": question, "context": format_context(retrieved)}
    )
    print(f"  检索到 {len(retrieved)} 段，提示词长度 {len(str(prompt_value))} 字符")
    for i, d in enumerate(retrieved):
        print(f"    [{i + 1}] {d.metadata['source']}")


def demo_real_world_mapping() -> None:
    hr("3. 真实项目里这三步换成什么")
    rows = [
        ("手写切分器", "langchain_text_splitters.RecursiveCharacterTextSplitter"),
        ("手写 BM25", "chromadb / faiss-cpu 向量库 + Embeddings 模型"),
        ("手写 Document", "langchain_community 的各种 Loader（PDF/网页/数据库）"),
        ("BaseRetriever 子类", "Chroma / FAISS 自带的 Retriever（接口一模一样）"),
        ("console 打印", "LangSmith 追踪（装 langsmith 就能用，已装）"),
    ]
    print(f"  {'本 demo 手写的':<22}{'真实项目里换成'}")
    print("  " + "-" * 78)
    for a, b in rows:
        print(f"  {a:<22}{b}")


def main() -> None:
    parser = argparse.ArgumentParser(description="手写 RAG")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("08", "不装任何库，手写一个 RAG")
    show_env_status()

    docs = build_documents()
    demo_retrieval(docs)

    model = resolve_model(
        fake=args.fake,
        responses=["根据资料，checkpointer 用于保存状态快照 [3]。"] * 5,
    )
    demo_rag_chain(model, docs)
    demo_real_world_mapping()

    hr("demo 08 结束")


if __name__ == "__main__":
    main()
