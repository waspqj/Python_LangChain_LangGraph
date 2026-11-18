"""demo 06 · LangGraph 基础：把流程画成图

覆盖的知识点：
    1. StateGraph + TypedDict 状态定义
    2. 节点（node）返回 dict = 对状态做**增量更新**
    3. 普通边 与 条件边（add_conditional_edges）
    4. 循环，以及 recursion_limit（递归上限）
    5. 并行分支（一个节点连到多个节点，多个节点汇到一个节点）
    6. reducer：多个节点写同一个字段时怎么合并（Annotated + operator.add）

心智模型：
    LangGraph = 一个有状态的图。
    节点之间传递的是"状态字典"，每个节点只负责改自己关心的那几个 key。

运行：
    python 06_langgraph_basics.py
"""

from __future__ import annotations

import argparse
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 1. 最简单的图：两个节点串起来
# ---------------------------------------------------------------------------
class SimpleState(TypedDict):
    name: str
    greeting: str


def demo_simple_graph(model) -> None:
    hr("1. 最简单的图")

    def greet(state: SimpleState) -> dict:
        # 只返回要改的 key，不用把整个 state 搬一遍
        return {"greeting": f"你好，{state['name']}！"}

    builder = StateGraph(SimpleState)
    builder.add_node("greet", greet)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    graph = builder.compile()

    result = graph.invoke({"name": "小王"})
    print("  最终状态 :", result)


# ---------------------------------------------------------------------------
# 2. 条件边 + 循环
# ---------------------------------------------------------------------------
class ReviewState(TypedDict):
    topic: str
    draft: str
    score: int
    rounds: Annotated[list[str], operator.add]   # reducer：每次返回都追加而不是覆盖


def demo_conditional_loop(model) -> None:
    """
    写 -> 评分 -> 不合格就回到"写"，合格就结束。

    循环必须有 termination 条件，否则会一直转下去；
    即使有条件，LangGraph 也有 recursion_limit 兜底。
    """
    hr("2. 条件边 + 循环")

    def write(state: ReviewState) -> dict:
        attempt = len(state.get("rounds", []))
        print(f"    [write] 第 {attempt + 1} 次起草：{state['topic']}")
        # 第一版平平无奇，第二版加上"深度"两个字，评分就能过
        if attempt == 0:
            return {"draft": f"关于{state['topic']}的一段草稿。"}
        return {"draft": f"关于{state['topic']}的一段有深度的草稿。"}

    def review(state: ReviewState) -> dict:
        # 这里用固定规则模拟评分：含"深度"二字就算高分
        score = 9 if "深度" in state.get("draft", "") else 4
        print(f"    [review] 评分 {score}")
        return {"score": score, "rounds": [f"第 {len(state.get('rounds', [])) + 1} 轮"]}

    builder = StateGraph(ReviewState)
    builder.add_node("write", write)
    builder.add_node("review", review)
    builder.add_edge(START, "write")
    builder.add_edge("write", "review")
    # 条件边：根据 review 的结果决定回到 write 还是结束
    builder.add_conditional_edges(
        "review",
        lambda state: "revise" if state["score"] < 6 else END,
        {"revise": "write", END: END},
    )
    graph = builder.compile()

    result = graph.invoke({"topic": "LangGraph", "draft": "", "score": 0, "rounds": []})
    print("  最终状态 :", result)

    hr("2b. recursion_limit：死循环是怎么被拦住的")

    # 造一个永远不会停的图
    def loop_forever(state: ReviewState) -> dict:
        return {"rounds": ["转一圈"]}

    bad = StateGraph(ReviewState)
    bad.add_node("loop", loop_forever)
    bad.add_edge(START, "loop")
    bad.add_edge("loop", "loop")
    bad_graph = bad.compile()

    try:
        bad_graph.invoke({"topic": "x", "draft": "", "score": 0, "rounds": []},
                         {"recursion_limit": 5})
    except Exception as exc:
        print(f"  被拦下了 : {type(exc).__name__}")
        print("  ", str(exc)[:130])
        print("  提示：LangGraph 默认 recursion_limit 很大（1.x 是 10000+），")
        print("       所以自己写的循环一定要有 termination 条件。")


# ---------------------------------------------------------------------------
# 3. 并行分支（fan-out / fan-in）
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    question: str
    web_result: str
    book_result: str
    summary: str


def demo_parallel(model) -> None:
    """一个节点连到两个节点，两个节点都跑完后汇到一个节点。"""
    hr("3. 并行分支")

    def fan_out(state: ResearchState) -> dict:
        print("    [fan_out] 分派任务")
        return {}

    def web(state: ResearchState) -> dict:
        print("    [web] 查网络")
        return {"web_result": "网络资料：LangGraph 是有状态图编排框架"}

    def book(state: ResearchState) -> dict:
        print("    [book] 查文献")
        return {"book_result": "文献资料：LangGraph 基于 Pregel 论文"}

    def merge(state: ResearchState) -> dict:
        print("    [merge] 汇总")
        return {"summary": state["web_result"] + " | " + state["book_result"]}

    builder = StateGraph(ResearchState)
    builder.add_node("fan_out", fan_out)
    builder.add_node("web", web)
    builder.add_node("book", book)
    builder.add_node("merge", merge)

    builder.add_edge(START, "fan_out")
    builder.add_edge("fan_out", "web")     # web   \
    builder.add_edge("fan_out", "book")    # book  > 两个分支并行
    builder.add_edge("web", "merge")       # merge /
    builder.add_edge("book", "merge")
    graph = builder.compile()

    result = graph.invoke({"question": "LangGraph 是什么", "web_result": "", "book_result": "", "summary": ""})
    print("  最终状态 :", result)


# ---------------------------------------------------------------------------
# 4. reducer：多节点写同一个字段
# ---------------------------------------------------------------------------
class CollectState(TypedDict):
    logs: Annotated[list[str], operator.add]   # 关键：带 reducer
    final: str


def demo_reducer(model) -> None:
    """
    没有 reducer 时，后写的节点会**覆盖**先写的；
    有了 reducer（这里是 list + operator.add），就是**追加**。
    """
    hr("4. reducer：追加而不是覆盖")

    def a(state: CollectState) -> dict:
        return {"logs": ["A 的日志"]}

    def b(state: CollectState) -> dict:
        return {"logs": ["B 的日志"]}

    builder = StateGraph(CollectState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    result = graph.invoke({"logs": [], "final": ""})
    print("  logs 字段 :", result["logs"])
    print("  结论：两个节点各写一次，reducer 让它们合并成 ['A 的日志', 'B 的日志']")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 基础")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("06", "LangGraph 基础：把流程画成图")
    show_env_status()

    model = resolve_model(fake=args.fake, responses=["预设回答" for _ in range(20)])

    demo_simple_graph(model)
    demo_conditional_loop(model)
    demo_parallel(model)
    demo_reducer(model)

    hr("demo 06 结束")


if __name__ == "__main__":
    main()
