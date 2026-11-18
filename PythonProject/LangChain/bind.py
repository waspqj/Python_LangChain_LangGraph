from langchain_deepseek import ChatDeepSeek
from pydantic import Field, BaseModel
from langchain_core.tools import tool


# 定义模型
model = ChatDeepSeek(model="deepseek-flash")


class MultiplyInput(BaseModel):
    """Multiply two integers"""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

@tool(args_schema = MultiplyInput)
def Multiply(a: int, b: int) -> int:
    return a * b

