"""demo 07 · LangGraph 的记忆与人工介入

覆盖的知识点：
    1. checkpointer：把每一步的状态快照存下来
    2. thread_id：一份 checkpointer 存很多条会话，用 thread_id 区分
    3. ``graph.get_state(config)`` 看当前快照
    4. ``graph.get_state_history(config)`` 看全部历史快照（time travel 的基础）
    5. ``interrupt()``：在图中间停下来，等人工给个答案再继续
    6. ``Command(resume=...)``：把人工的答案喂回去，图继续往下跑

本机没装 langgraph-checkpoint-sqlite，所以只能用内存版：
状态存在内存里，进程一退就没了。真实项目请换 SqliteSaver / PostgresSaver。

运行：
    python 07_langgraph_memory_hitl.py
"""

from __future__ import annotations

import argparse
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    question: str
    answer: str
    approved: str


# ---------------------------------------------------------------------------
# 1. checkpointer + thread_id
# ---------------------------------------------------------------------------
def demo_checkpoint(model) -> None:
    hr("1. checkpointer 与 thread_id")

    def answer_node(state: ChatState) -> dict:
        # 真实项目这里会调模型；这里用规则模拟，好把注意力放在状态上
        return {"answer": f"（模拟回答）{state['question']} 的答案是 42"}

    builder = StateGraph(ChatState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    # 同一个 thread_id = 同一条会话
    cfg_a = {"configurable": {"thread_id": "会话A"}}
    graph.invoke({"question": "第一个问题", "answer": "", "approved": ""}, cfg_a)
    graph.invoke({"question": "第二个问题", "answer": "", "approved": ""}, cfg_a)

    snapshot = graph.get_state(cfg_a)
    print(f"  当前快照的 values : {snapshot.values}")
    print(f"  当前快照的 next   : {snapshot.next}")
    print(f"  会话A 的历史条数   : {len(list(graph.get_state_history(cfg_a)))}")

    # 换一个 thread_id = 另一条会话，历史互不相通
    cfg_b = {"configurable": {"thread_id": "会话B"}}
    graph.invoke({"question": "B 的问题", "answer": "", "approved": ""}, cfg_b)
    snap_b = graph.get_state(cfg_b)
    print(f"  会话B 的 values    : {snap_b.values}")
    print("  结论：同一个 checkpointer 靠 thread_id 隔离出多条独立会话")


# ---------------------------------------------------------------------------
# 2. time travel：回到历史某一步
# ---------------------------------------------------------------------------
def demo_time_travel(model) -> None:
    """
    每个节点执行完都会存一份快照，可以用快照里的 config 回到那一步重跑。
    """
    hr("2. time travel：回到历史某一步")

    def step(state: ChatState) -> dict:
        return {"answer": f"第 {len(state.get('answer', '')) + 1} 版答案"}

    builder = StateGraph(ChatState)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "回放"}}
    graph.invoke({"question": "q", "answer": "", "approved": ""}, cfg)

    history = list(graph.get_state_history(cfg))
    print(f"  历史快照条数 : {len(history)}")
    for snap in history[:3]:
        print(f"    values={snap.values}  next={snap.next}")

    if len(history) > 1:
        # 用某个历史快照的 config 重新触发，图会从那一步接着跑
        past = history[-1]
        print("  选用最早的那个快照重跑：")
        graph.invoke(None, past.config)
        print(f"  重跑后 values : {graph.get_state(cfg).values}")


# ---------------------------------------------------------------------------
# 3. interrupt：人工介入
# ---------------------------------------------------------------------------
def demo_interrupt(model) -> None:
    """
    典型场景：高风险操作（发邮件、付款、删数据）先让人确认一下。

    interrupt() 会把图**暂停**在这里，等你用 Command(resume=...) 送一个值回来。
    """
    hr("3. interrupt：人工介入")

    def risky_node(state: ChatState) -> dict:
        # 图停在这一行，等外部送一个答案回来
        approval = interrupt(
            {
                "action": "发送邮件",
                "content": state["question"],
                "prompt": "请确认是否发送（同意 / 拒绝）",
            }
        )
        return {"approved": approval}

    builder = StateGraph(ChatState)
    builder.add_node("risky", risky_node)
    builder.add_edge(START, "risky")
    builder.add_edge("risky", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "人工确认"}}

    # 第一次调用：跑到 risky 节点就停住
    result = graph.invoke({"question": "给老板发个消息", "answer": "", "approved": ""}, cfg)
    print(f"  第一次 invoke 返回的 keys : {list(result.keys())}")

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        payload = interrupts[0].value
        print(f"  图停在节点 '{interrupts[0].id}'，等待人工处理")
        print(f"  中断载荷 : {payload}")

    # 第二次调用：把人工的答案送回去，图继续往下跑
    final = graph.invoke(Command(resume="同意，已发送"), cfg)
    print(f"  继续执行后 approved = {final['approved']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆与人工介入")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("07", "LangGraph 的记忆与人工介入")
    show_env_status()

    model = resolve_model(fake=args.fake, responses=["预设回答" for _ in range(10)])

    demo_checkpoint(model)
    demo_time_travel(model)
    demo_interrupt(model)

    hr("demo 07 结束")


if __name__ == "__main__":
    main()
