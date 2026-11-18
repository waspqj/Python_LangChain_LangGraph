"""demo 02 · 提示词模板与 LCEL 链式编排

覆盖的知识点：
    1. ChatPromptTemplate：变量填充、多角色消息、MessagesPlaceholder
    2. ``|`` 管道：把 prompt / model / parser 串成一条链
    3. 核心 Runnable 组件：
       RunnablePassthrough / RunnableLambda / RunnableParallel / RunnableBranch
    4. 把子链当成一个步骤复用（链的组合）

版本提醒：LangChain 1.x 里 ``langchain.prompts`` 已经没有���
必须从 ``langchain_core.prompts`` 导入。

运行：
    python 02_prompt_lcel.py            # 用真实模型
    python 02_prompt_lcel.py --fake     # 离线跑通流程
"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from common import banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 1. 提示词模板：变量填充
# ---------------------------------------------------------------------------
def demo_prompt(model) -> None:
    """
    ChatPromptTemplate 只做一件事：把 ``{变量}`` 填进模板。

    和老版本 ``langchain.prompts`` 的区别：1.x 里必须从 langchain_core 导入。
    """
    hr("1. ChatPromptTemplate")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}，请用{lang}回答。"),
            ("human", "{question}"),
        ]
    )
    print("  模板变量 :", prompt.input_variables)
    print("  填完之后 :", prompt.format(role="历史学家", lang="中文", question="简述秦朝"))

    # 模板本身也是 Runnable，可以直接接模型
    chain = prompt | model | StrOutputParser()
    print("  链式结果 :", chain.invoke({"role": "历史学家", "lang": "中文", "question": "简述秦朝"}))


# ---------------------------------------------------------------------------
# 2. MessagesPlaceholder：把已有消息插进模板
# ---------------------------------------------------------------------------
def demo_placeholder(model) -> None:
    """带历史对话时，用 MessagesPlaceholder 留一个"消息列表"的坑。"""
    hr("2. MessagesPlaceholder")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是助手。"),
            MessagesPlaceholder("history"),   # 这个坑会被一整串历史消息填上
            ("human", "{question}"),
        ]
    )
    print("  模板变量 :", prompt.input_variables)

    from langchain_core.messages import AIMessage, HumanMessage

    filled = prompt.invoke(
        {
            "history": [
                HumanMessage(content="我叫小王"),
                AIMessage(content="你好小王"),
            ],
            "question": "我叫什么？",
        }
    )
    print("  拼好的消息列表:")
    for m in filled.messages:
        print(f"    [{m.type}] {m.content}")


# ---------------------------------------------------------------------------
# 3. RunnablePassthrough.assign：往输入里加字段
# ---------------------------------------------------------------------------
def demo_passthrough(model) -> None:
    """
    ``|`` 管道左边传下来的东西，会一路流到右边。
    assign 可以在中间"加一个字段"再往下传。
    """
    hr("3. RunnablePassthrough.assign")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是助手。"),
            ("human", "问题：{question}\n补充：{extra}"),
        ]
    )

    def make_extra(inputs: dict) -> str:
        return f"（这条补充信息是程序自动加的，长度 {len(inputs['question'])}）"

    chain = (
        RunnablePassthrough.assign(extra=RunnableLambda(make_extra))
        | prompt
        | model
        | StrOutputParser()
    )

    print("  结果 :", chain.invoke({"question": "什么是 LCEL"}))
    print("  提示：assign 不改变已有字段，只是往字典里追加 key")


# ---------------------------------------------------------------------------
# 4. RunnableParallel：一次并行跑多条支线
# ---------------------------------------------------------------------------
def demo_parallel(model) -> None:
    """parallel 里每个分支收到的是同一份输入，结果合成一个 dict。"""
    hr("4. RunnableParallel")

    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是{role}。"), ("human", "{topic}")]
    )

    # 每个分支收到的是同一份输入 dict，结果合成一个 dict 返回
    # partial() 用来"预先填好"部分变量，剩下的运行时再填
    chain = RunnableParallel(
        技术=prompt.partial(role="技术专家") | model | StrOutputParser(),
        字数=RunnableLambda(lambda x: len(x["topic"])),
    )
    print("  并行结果 :", chain.invoke({"topic": "西湖"}))


# ---------------------------------------------------------------------------
# 5. RunnableBranch：按条件选支线
# ---------------------------------------------------------------------------
def demo_branch(model) -> None:
    """if/elif/else 的 Runnable 版本。"""
    hr("5. RunnableBranch")

    def is_question(x: dict) -> bool:
        return "?" in x["text"] or "？" in x["text"]

    # RunnableBranch(*branches)：
    #   前面的每个元素是 (条件, Runnable) 二元组；
    #   **最后一个元素直接就是默认 Runnable**（不要再包成二元组）。
    branch = RunnableBranch(
        (is_question, RunnableLambda(lambda x: f"[走问句分支] {x['text']}")),
        RunnableLambda(lambda x: f"[走默认分支] {x['text']}"),
    )

    print("  ", branch.invoke({"text": "今天天气怎么样？"}))
    print("  ", branch.invoke({"text": "讲个笑话"}))


# ---------------------------------------------------------------------------
# 6. 链的组合：把一条链当成另一个步骤
# ---------------------------------------------------------------------------
def demo_compose(model) -> None:
    """
    ``Runnable`` 本身也是 Runnable，所以链可以嵌链。
    这是 LangChain 1.x（LCEL）最核心的组合能力。
    """
    hr("6. 链的组合")

    清洗 = RunnableLambda(lambda x: {"topic": x["topic"].strip().upper()})
    提问 = ChatPromptTemplate.from_messages(
        [("human", "请用一句话解释：{topic}")]
    )
    成文 = (
        RunnablePassthrough.assign(
            摘要=RunnableLambda(lambda x: f"主题是 {x['topic']}，共 {len(x['topic'])} 字符")
        )
        | 提问
        | model
        | StrOutputParser()
    )

    总链 = 清洗 | 成文
    print("  结果 :", 总链.invoke({"topic": "  langgraph  "}))


# ---------------------------------------------------------------------------
# 7. batch：一条链批量跑
# ---------------------------------------------------------------------------
def demo_batch(model) -> None:
    hr("7. batch")

    prompt = ChatPromptTemplate.from_messages([("human", "给{word}造个比喻")])
    chain = prompt | model | StrOutputParser()

    outs = chain.batch([{"word": "代码"}, {"word": "生活"}])
    for o in outs:
        print("  ", o)


def main() -> None:
    parser = argparse.ArgumentParser(description="提示词与 LCEL")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("02", "提示词模板与 LCEL 链式编排")
    show_env_status()

    model = resolve_model(fake=args.fake, responses=["这是一段预设的回答。" for _ in range(30)])
    if args.fake:
        hr("离线模式：使用假模型，输出为预设内容")

    demo_prompt(model)
    demo_placeholder(model)
    demo_passthrough(model)
    demo_parallel(model)
    demo_branch(model)
    demo_compose(model)
    demo_batch(model)

    hr("demo 02 结束")


if __name__ == "__main__":
    main()
