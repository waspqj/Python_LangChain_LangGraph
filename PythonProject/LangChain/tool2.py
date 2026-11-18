from langchain_core.tools import StructuredTool
from typing import List, Tuple

# 使用StructuredTool类提供的函数创建工具

# 1. 常规用法（函数/工具的描述也是必不可少）
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

calculator_tool = StructuredTool.from_function(func=add)
print(calculator_tool.invoke({"a": 1, "b": 2}))

# 2. 依赖Pydantic类，函数不提供描述，通过参数配置
from pydantic import BaseModel, Field

# class CalculatorInput(BaseModel):
#     a: int = Field(description="first number")
#     b: int = Field(description="second number")
#
# def multiply(a:int, b:int) -> int:
#     return a * b
#
# calculator_tool = StructuredTool.from_function(
#     func=multiply,
#     name="Calculator",
#     description="两数相除",
#     args_schema=CalculatorInput,
# )
#
# print(calculator_tool.invoke({"a": 1, "b": 2}))
# print(calculator_tool.name)
# print(calculator_tool.description)
# print(calculator_tool.args)
# # print(calculator_tool.args_schema)

# 加入response_format配置 —— 保留artifact

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

def multiply(a:int, b:int) -> Tuple[str, List[int]]:
    nums = [a, b]
    content = f"{nums}相乘的结果是{a * b}"
    return content, nums

calculator_tool = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="两数相除",
    args_schema=CalculatorInput,
    response_format="content_and_artifact"  #保留artifact部分
)

# 普通调用，只会返回content
print(calculator_tool.invoke({"a":2, "b": 3}))
# 模拟大模型调用工具
print(calculator_tool.invoke(
    {
        "name": "Calculator",
        "args": {"a": 2, "b": 3},
        "id": "123", # 必须的,与工具调用关联的标识符，将请求和结果关联
        "type": "tool_call", # 必须
    }
))
# result：“content='[2, 3]相乘的结果是6' name='Calculator' tool_call_id='123' artifact=[2, 3]“

