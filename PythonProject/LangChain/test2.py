from langchain_openai import ChatOpenAI

# 使用链式执行
# 定义大模型
model = ChatOpenAI(
    model="MiniMax-M3"
)
# 定义消息
from langchain_core.messages import SystemMessage, HumanMessage
message = [
    SystemMessage(content="你是一个专业的心理医生，患者对你说了如下的话，你会怎么回答(用中文回复"),
    HumanMessage(content="I‘m tired")
]
# 定义输出解析器
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# 定义链
# chain = model | parser

# 用RunnableSequence定义链
#from langchain_core.runnables import RunnableSequence
#chain = RunnableSequence(first = model, last = parser)

# 用pipe 替换 |
chain = model.pipe(parser)

# 链式执行
print(chain.invoke(message))
