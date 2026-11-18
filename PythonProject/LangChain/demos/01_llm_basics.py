"""demo 01 · 大模型基础调用

覆盖的知识点：
    1. invoke / stream / batch 三种调用方式
    2. 消息类型：SystemMessage / HumanMessage / AIMessage
    3. 模型参数：temperature / max_tokens，以及用 bind() 临时改参数
    4. 从 AIMessage 里读 token 用量

运行：
    python 01_llm_basics.py            # 用真实模型
    python 01_llm_basics.py --fake     # 离线跑通流程，不消耗额度
"""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common import banner, count_tokens, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 1. invoke：最朴素的调用方式
# ---------------------------------------------------------------------------
def demo_invoke(model) -> None:
    """传消息列表，拿回一条 AIMessage。"""
    messages = [
        SystemMessage(content="你是一个简洁的中文助手，回答控制在 30 字以内。"),
        HumanMessage(content="用一句话解释什么是 LangChain。"),
    ]

    result = model.invoke(messages)

    print(f"  返回类型 : {type(result).__name__}")
    print(f"  content  : {result.content}")
    print(f"  token 用量: {getattr(result, 'usage_metadata', None)}")


# ---------------------------------------------------------------------------
# 2. 纯字符串输入：LangChain 会自动包成 HumanMessage
# ---------------------------------------------------------------------------
def demo_string_input(model) -> None:
    """invoke("...") 等价于 invoke([HumanMessage("...")])。"""
    result = model.invoke("用一句话说清楚 LangChain 和 LangGraph 的区别")
    print(f"  content  : {result.content}")
    print(f"  是 AIMessage 吗: {isinstance(result, AIMessage)}")


# ---------------------------------------------------------------------------
# 3. stream：流式输出
# ---------------------------------------------------------------------------
def _chunk_text(chunk) -> str:
    """AIMessageChunk.content 可能是 str，也可能是 list[dict]，都要兼容。"""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def demo_stream(model) -> None:
    """token 级返回，聊天界面必须用这个。"""
    print("  流式输出：", end="", flush=True)
    for chunk in model.stream("用一句话描述杭州"):
        print(_chunk_text(chunk), end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 4. batch：批量并发
# ---------------------------------------------------------------------------
def demo_batch(model) -> None:
    """一次提交多条输入，框架内部帮你并发（比 for 循环调 invoke 快得多）。"""
    inputs = [
        "1+1 等于几？只回答数字。",
        "中国的首都是哪里？只回答城市名。",
        "把 '早上好' 翻译成英文。",
    ]
    results = model.batch(inputs)
    for q, r in zip(inputs, results):
        print(f"  Q: {q}")
        print(f"  A: {r.content}")


# ---------------------------------------------------------------------------
# 5. 参数：temperature / max_tokens，以及 bind() 临时覆盖
# ---------------------------------------------------------------------------
def demo_params(model) -> None:
    """
    temperature 越高越发散（0 附近最稳定），max_tokens 卡住输出长度上限。

    model.bind(...) 返回一个"绑定了参数的新模型"，不改动原模型，
    这是 LangChain 里做"同一模型不同配置"的标准手法。
    """
    hr("模型参数：bind() 临时覆盖 max_tokens")

    loose = model.invoke("用 100 个字介绍杭州西湖")
    tight = model.bind(max_tokens=20).invoke("用 100 个字介绍杭州西湖")

    print(f"  不限制      : {count_tokens(str(loose.content)):>4} tokens")
    print(f"  max_tokens=20: {count_tokens(str(tight.content)):>4} tokens")
    print(f"  绑定后是同类型吗: {type(tight).__name__}")


# ---------------------------------------------------------------------------
# 6. 同一个模型，换个 SystemMessage 就是另一个"应用"
# ---------------------------------------------------------------------------
def demo_reusable(model) -> None:
    """
    模型本身不含人格，人格来自消息里的 SystemMessage。

    注意：``model.bind(system=...)`` 是**无效**的——ChatOpenAI / ChatDeepSeek
    都没有 system 字段，bind 进去的参数会被当成 API 扩展参数发出去。
    想固定人格，要么像这里手动塞 SystemMessage，
    要么用 demo 02 里的 ChatPromptTemplate。
    """
    hr("同一个模型，两个不同人格")

    strict = model.invoke(
        [
            SystemMessage(content="你只回答一个词，不要任何解释。"),
            HumanMessage(content="介绍你自己"),
        ]
    )
    friendly = model.invoke(
        [
            SystemMessage(content="你是一个热情的客服，回答里要有 emoji。"),
            HumanMessage(content="介绍你自己"),
        ]
    )

    print(f"  严格人格: {strict.content}")
    print(f"  热情人格: {friendly.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain 基础调用")
    parser.add_argument(
        "--fake", action="store_true", help="用假模型离线跑通流程，不消耗额度"
    )
    args = parser.parse_args()

    banner("01", "大模型基础调用")
    show_env_status()

    model = resolve_model(
        fake=args.fake,
        responses=[
            "LangChain 是一个把大模型和工具、数据源连接起来的开发框架。",
            "LangChain 负责编排，LangGraph 负责有状态、有记忆的流程编排。",
            "杭州",
            "2",
            "北京",
            "Good morning",
            "杭州西湖风景秀丽，是著名的旅游胜地。",
            "杭州",
            "杭州",
        ],
    )
    if args.fake:
        hr("离线模式：使用假模型，输出为预设内容")

    hr("1. invoke")
    demo_invoke(model)

    hr("2. 字符串输入")
    demo_string_input(model)

    hr("3. stream")
    demo_stream(model)

    hr("4. batch")
    demo_batch(model)

    hr("5. 参数")
    demo_params(model)

    hr("6. 复用")
    demo_reusable(model)

    hr("demo 01 结束")


if __name__ == "__main__":
    main()
