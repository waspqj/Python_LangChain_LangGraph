"""demo 04 · 工具（Tools）

覆盖的知识点：
    1. ``@tool`` 的三种定义方式：docstring / Pydantic args_schema / Annotated
    2. 工具对象长什么样：name / description / args（这就是发给模型的 JSON Schema）
    3. ``model.bind_tools(tools)`` 之后，模型返回的是带 ``tool_calls`` 的 AIMessage
    4. 手工实现工具调用循环 —— 这就是 Agent 的底层原理
    5. 工具报错时怎么把错误信息喂回模型

本机没装 langchain-community，所以这里的工具都是自己写的 Python 函数；
真实项目里还会有搜索引擎、数据库、浏览器等。

运行：
    python 04_tools.py            # 用真实模型
    python 04_tools.py --fake     # 用脚本化假模型离线跑通工具循环
"""

from __future__ import annotations

import argparse
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common import (
    ScriptedChatModel,
    banner,
    hr,
    resolve_model,
    show_env_status,
)


# ---------------------------------------------------------------------------
# 1. 工具的三种定义方式
# ---------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    return a + b


class MultiplyInput(BaseModel):
    """两个整数相乘。"""

    a: int = Field(..., description="第一个因数")
    b: int = Field(..., description="第二个因数")


@tool(args_schema=MultiplyInput)
def multiply(a: int, b: int) -> int:
    """两数相乘。"""
    return a * b


@tool
def divide(a: Annotated[float, "被除数"], b: Annotated[float, "除数"]) -> float:
    """两数相除。"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


TOOLS = [add, multiply, divide]


def demo_tool_objects() -> None:
    """
    工具最终就是一个"有名字、有描述、有入参 schema"的函数。

    name / description / args 这三样会被序列化后发给模型，
    模型就是靠它们决定"要不要调这个工具、传什么参数"。
    """
    hr("1. 工具对象长什么样")

    for t in TOOLS:
        print(f"  name        : {t.name}")
        print(f"  description : {t.description}")
        print(f"  args schema : {t.args}")
        print("  " + "-" * 60)

    print("  直接调用 add.invoke({'a': 1, 'b': 2}) =", add.invoke({"a": 1, "b": 2}))


# ---------------------------------------------------------------------------
# 2. bind_tools：模型开始"想调工具"
# ---------------------------------------------------------------------------
def demo_bind_tools(model) -> None:
    hr("2. bind_tools 之后模型返回什么")

    if hasattr(model, "reset"):
        model.reset()
    model_with_tools = model.bind_tools(TOOLS)
    ai_msg = model_with_tools.invoke([HumanMessage(content="帮我算 3 + 5")])

    print(f"  返回类型 : {type(ai_msg).__name__}")
    print(f"  content  : {ai_msg.content!r}")
    print(f"  tool_calls: {ai_msg.tool_calls}")
    if ai_msg.tool_calls:
        call = ai_msg.tool_calls[0]
        print(f"  第一个调用: name={call['name']} args={call['args']} id={call['id']}")


# ---------------------------------------------------------------------------
# 3. 手工工具调用循环（Agent 的底层）
# ---------------------------------------------------------------------------
def manual_tool_loop(model, question: str, max_turns: int = 5):
    """
    完整循环：

        模型 --(tool_calls)--> 执行工具 --(ToolMessage)--> 回灌给模型 --> 直到没有 tool_calls

    Agent 框架（demo 05）做的就是这件事，只不过用图来编排。
    """
    hr("3. 手工工具调用循环")

    if hasattr(model, "reset"):
        model.reset()
    tool_by_name = {t.name: t for t in TOOLS}
    model_with_tools = model.bind_tools(TOOLS)
    messages: list = [HumanMessage(content=question)]

    for turn in range(max_turns):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            print(f"  第 {turn + 1} 轮：模型给出最终回答 -> {ai_msg.content}")
            return ai_msg, messages

        for call in ai_msg.tool_calls:
            fn = tool_by_name.get(call["name"])
            print(f"  第 {turn + 1} 轮：模型请求调用 {call['name']}({call['args']})")
            try:
                result = fn.invoke(call["args"]) if fn else f"没有名为 {call['name']} 的工具"
            except Exception as exc:
                # 工具报错时，把错误当成"工具返回值"喂回模型，让模型自己改
                result = f"工具执行失败：{exc}"
                print(f"    -> 工具报错：{exc}")
            else:
                print(f"    -> 工具返回：{result}")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    print("  达到最大轮数，退出")
    return None, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="工具调用")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("04", "工具（Tools）")
    show_env_status()

    demo_tool_objects()

    if args.fake:
        hr("离线模式：用脚本化假模型模拟模型的 tool_calls")
        from langchain_core.messages import AIMessage

        model = ScriptedChatModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "args": {"a": 3, "b": 5}, "id": "call_1"}
                    ],
                ),
                AIMessage(content="3 + 5 = 8"),
            ]
        )
    else:
        model = resolve_model()

    demo_bind_tools(model)
    manual_tool_loop(model, "帮我算 3 + 5")

    hr("demo 04 结束")


if __name__ == "__main__":
    main()
