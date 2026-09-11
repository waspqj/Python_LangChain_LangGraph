from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sympy.functions.elementary import integers
from typing_extensions import Annotated


#使用@tool装饰器创建工具

# 1. 文档字符串,工具描述
# @tool
# def Add(a:int, b:int) -> int:
#     """Add two integers
#
#     Args:
#         a : first int
#         b : second int
#     """
#     return a + b
#
# print(Add.invoke({"a":2,"b":3}))

# 2. 依赖Pydantic类
class MultiplyInput(BaseModel):
    """Multiply two integers"""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

@tool(args_schema = MultiplyInput)
def Multiply(a: int, b: int) -> int:
    return a * b

print(Multiply.invoke({"a": 1, "b": 2}))

# 3. 依赖Annotated
@tool
def add(
        a: Annotated[int, ..., "First integer"],
        b: Annotated[int, ..., "Second integer"]
) -> int:
    """Add two integers"""
    return a + b

print(add.invoke({"a": 1, "b": 2}))