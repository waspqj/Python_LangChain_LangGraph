from langchain.chat_models import init_chat_model
from sympy.physics.units import temperature

# 使用 init_chat_model 创建聊天模型
# minimax_model = init_chat_model("MiniMax-M3", model_provider="minimax",
#                                 temperature = 0,)
# deepseek_model = init_chat_model("deepseek-flash", model_provider="deepseek",
#                                 temperature = 0,)
#
# print(deepseek_model.invoke("what's your name?").content)

# # 创建可配置模型
# configurable_model = init_chat_model(temperature=1,)
# # 动态修改配置, 初始化模型并调用
# print(configurable_model.invoke("what's your name?", config={"configurable": {"model": "deepseek-flash"}}).content)

# 创建有默认值的可配置模型
configurable_model = init_chat_model(
    model="deepseek-v4-pro",
    temperature=2,
    max_tokens=100000,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens",),
    config_prefix="first"
)
# 使用默认配置
# print(configurable_model.invoke("描写青草").content + "\n\n\n")
# 动态修改配置
print(configurable_model.invoke(
    input="描写青草",
    config={
        "configurable": {
            "first_model": "deepseek-flash",
            "first_temperature": 2,
            "first_max_tokens": 30,
        }
    },).content)
