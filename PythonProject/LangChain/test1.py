from langchain_openai import ChatOpenAI

# 直接定义并调用大模型
# 定义大模型
model = ChatOpenAI(
    model="MiniMax-M3",
    temperature=1,
    max_tokens=100,
)
# 定义消息列表
from langchain_core.messages import HumanMessage, SystemMessage
message = [
    SystemMessage(content="Translate the following from Chinese into English"),
    HumanMessage(content="枯藤老树昏鸦"),
]
# 调用大模型
result = model.invoke(message)
# print(result)

# 输出解析
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
print(parser.invoke(result))


# print(model.invoke("‘小猫团着睡觉，像__’续写").content)