"""demo 03 · 结构化输出（让模型吐出 JSON / Pydantic 对象）

覆盖的知识点：
    1. ``with_structured_output(Schema)``  —— 1.x 推荐写法，模型直接被约束
    2. ``PydanticOutputParser``          —— 把格式说明塞进提示词，再解析
    3. ``JsonOutputParser``              —— 只保证是合法 JSON，不保证字段
    4. Pydantic 里的 Field(description=...) 会直接影响模型输出质量

三种方式的区别：
    with_structured_output  >  PydanticOutputParser  >  JsonOutputParser
    前者由模型侧约束（走 tool calling / json schema），最稳；
    后者只是"提示模型请输出 JSON，然后我再解析"，模型不听话就会解析失败。

运行：
    python 03_structured_output.py            # 用真实模型
    python 03_structured_output.py --fake     # 离线跑通 PydanticOutputParser
"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import banner, hr, resolve_model, show_env_status


# ---------------------------------------------------------------------------
# 目标结构
# ---------------------------------------------------------------------------
class MovieReview(BaseModel):
    """一条电影短评的结构化结果。"""

    title: str = Field(description="电影的中文名")
    score: int = Field(description="0 到 10 之间的整数评分")
    tags: list[str] = Field(default_factory=list, description="不超过 3 个的关键词标签")
    recommend: bool = Field(description="是否推荐观看")


# ---------------------------------------------------------------------------
# 方式一：with_structured_output（推荐）
# ---------------------------------------------------------------------------
def demo_with_structured_output(model) -> None:
    """
    模型被约束成"只能产出符合 MovieReview 的 JSON"，直接返回 Pydantic 对象。

    注意：底层要走 tool calling，所以**假模型不支持**，必须用真实模型。
    """
    hr("1. with_structured_output（需要真实模型）")

    try:
        structured = model.with_structured_output(MovieReview)
    except NotImplementedError:
        print("  当前模型不支持工具调用（离线假模型），跳过。")
        print("  用真实模型跑本 demo 即可看到效果。")
        return

    result = structured.invoke("给电影《活着》打个分并给出标签")
    print(f"  返回类型 : {type(result).__name__}")
    print(f"  字段     : title={result.title} score={result.score} "
          f"tags={result.tags} recommend={result.recommend}")
    print(f"  重新序列化: {result.model_dump_json()}")


# ---------------------------------------------------------------------------
# 方式二：PydanticOutputParser
# ---------------------------------------------------------------------------
def demo_pydantic_output_parser(model) -> None:
    """把 Pydantic 的 JSON Schema 说明塞进 system 提示词，模型照着输出，再解析。"""
    hr("2. PydanticOutputParser")

    parser = PydanticOutputParser(pydantic_object=MovieReview)

    # get_format_instructions() 就是那段"请严格按此 JSON schema 输出"的说明
    fmt = parser.get_format_instructions()
    print("  格式说明前 120 字:", fmt[:120].replace("\n", " "), "...")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是电影评论助手。\n{format_instructions}"),
            ("human", "请评价电影《{movie}》。"),
        ]
    ).partial(format_instructions=fmt)

    chain = prompt | model | parser

    try:
        result = chain.invoke({"movie": "活着"})
    except Exception as exc:
        print(f"  解析失败（模型没按格式输出）：{type(exc).__name__}")
        print("  这就是 PydanticOutputParser 的软肋 —— 靠提示词约束，不是硬约束。")
        return

    print(f"  返回类型 : {type(result).__name__}")
    print(f"  内容     : {result}")


# ---------------------------------------------------------------------------
# 方式三：JsonOutputParser
# ---------------------------------------------------------------------------
def demo_json_output_parser(model) -> None:
    """只保证是合法 JSON，字段不管。"""
    hr("3. JsonOutputParser")

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是助手。请只输出 JSON，不要任何解释。\n{format_instructions}"),
            ("human", "用 JSON 描述杭州西湖。"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    try:
        result = chain.invoke({})
    except Exception as exc:
        print(f"  解析失败：{type(exc).__name__}")
        return

    print(f"  返回类型 : {type(result).__name__}")
    print(f"  内容     : {result}")


# ---------------------------------------------------------------------------
# 对比表
# ---------------------------------------------------------------------------
def demo_compare() -> None:
    hr("4. 三种方式对比")
    rows = [
        ("with_structured_output", "模型侧约束(tool calling)", "最强", "需真实模型"),
        ("PydanticOutputParser", "提示词约束 + 后置解析", "较强", "模型不听话会报错"),
        ("JsonOutputParser", "提示词约束 + 后置解析", "弱", "字段完全不保证"),
    ]
    print(f"  {'方式':<26}{'约束方式':<28}{'稳定性':<8}备注")
    print("  " + "-" * 78)
    for a, b, c, d in rows:
        print(f"  {a:<26}{b:<28}{c:<8}{d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="结构化输出")
    parser.add_argument("--fake", action="store_true", help="离线模式")
    args = parser.parse_args()

    banner("03", "结构化输出")
    show_env_status()

    # 假模型必须吐出符合 schema 的 JSON，否则 PydanticOutputParser 会解析失败
    fake_json = (
        '{"title": "活着", "score": 9, "tags": ["年代", "命运"], "recommend": true}'
    )
    model = resolve_model(
        fake=args.fake,
        responses=[fake_json, fake_json, fake_json, fake_json],
    )

    demo_with_structured_output(model)
    demo_pydantic_output_parser(model)
    demo_json_output_parser(model)
    demo_compare()

    hr("demo 03 结束")


if __name__ == "__main__":
    main()
