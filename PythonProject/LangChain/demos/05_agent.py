"""demo 05 · Agent（智能体）

覆盖的知识点：
    1. ``create_agent(model, tools, system_prompt=...)`` —— LangChain 1.x 的新 Agent 入口
    2. Agent 返回的是消息列表，中间过程（工具调用）都在里面
    3. 记忆：``checkpointer=InMemorySaver()`` + ``thread_id``
    4. middleware：在模型调用前后挂钩子

要点：LangChain 1.x 的 ``create_agent`` 底层就是 LangGraph 图，
所以 Agent 天生带持久化、流式、中断能力 —— 这也是 1.x 相比 0.x 最大的变化。

运行：
    python 05_agent.py            # 用真实模型
    python 05_agent.py --fake     # 用脚本化假模型离线跑通
"""

from __future__ import annotations

import argparse
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common import ScriptedChatModel, banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    return a + b


@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。

    Args:
        city: 城市名，例如 杭州
    """
    fake = {"杭州": "晴，26℃", "北京": "多云，18℃", "上海": "小雨，22℃"}
    return fake.get(city, f"没有 {city} 的天气数据")


TOOLS = [add, get_weather]


# ---------------------------------------------------------------------------
# 1. 最基础的 Agent
# ---------------------------------------------------------------------------
def demo_basic_agent(model) -> None:
    hr("1. create_agent 基础用法")

    agent = create_agent(
        model,
        tools=TOOLS,
        system_prompt="你是一个计算助手，会调用工具回答问题。",
    )

    result = agent.invoke({"messages": [HumanMessage(content="帮我算 3 + 5")]})

    print("  返回的 state keys:", list(result.keys()))
    print("  消息列表:")
    for m in result["messages"]:
        print(f"    [{m.type}] {m.content}")


# ---------------------------------------------------------------------------
# 2. 记忆：checkpointer + thread_id
# ---------------------------------------------------------------------------
def demo_agent_memory(model) -> None:
    """
    同一个 thread_id = 同一段对话历史；换 thread_id 就是另开一个会话。

    没装 langgraph-checkpoint-sqlite，所以只能用内存版：
    进程一退历史就没了（真实项目请换成 SqliteSaver）。
    """
    hr("2. 记忆：checkpointer + thread_id")

    agent = create_agent(
        model,
        tools=TOOLS,
        system_prompt="你是一个乐于助人的助手。",
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "会话A"}}

    agent.invoke({"messages": [HumanMessage(content="我叫小王")]}, config)
    print("  第一轮已发送")

    result = agent.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, config)

    print("  第二轮返回:")
    for m in result["messages"]:
        print(f"    [{m.type}] {m.content}")

    # 换一个 thread_id = 另一个会话，历史互不相通
    other = {"configurable": {"thread_id": "会话B"}}
    result_b = agent.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, other)
    print(f"  换个 thread_id 后，历史消息条数: {len(result_b['messages'])}（全新会话）")


# ---------------------------------------------------------------------------
# 3. middleware：在模型调用前挂钩子
# ---------------------------------------------------------------------------
def demo_middleware(model) -> None:
    """
    middleware 可以在模型调用前后插入自定义逻辑：
    打印日志、裁剪历史、注入上下文、加护栏，都靠它。
    """
    hr("3. middleware")

    @before_model
    def log_before_model(state, runtime):
        # 返回 None 表示不修改状态
        print(f"  [middleware] 模型即将被调用，当前消息数 = {len(state['messages'])}")
        return None

    agent = create_agent(
        model,
        tools=TOOLS,
        system_prompt="你是一个助手。",
        middleware=[log_before_model],
    )

    agent.invoke({"messages": [HumanMessage(content="你好")]})
    print("  [middleware] 已插入")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("05", "Agent（智能体）")
    show_env_status()

    if args.fake:
        hr("离线模式：脚本化假模型")
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

    demo_basic_agent(model)
    demo_agent_memory(model)
    demo_middleware(model)

    hr("demo 05 结束")


if __name__ == "__main__":
    main()
