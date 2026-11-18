"""demo 公共模块：环境配置 + 模型工厂 + 离线自测。

本机现有环境（**不需要再装任何包**）已经覆盖：
    langchain 1.4.0           langchain-core 1.6.2       langchain-openai 1.6.2
    langchain-deepseek 1.1.0  langgraph 1.2.11           langgraph-checkpoint 4.2.0
    langgraph-prebuilt 1.1.0  langsmith 0.12.4           python-dotenv

三个约定：
    1. 密钥只从 .env 读，绝不写进代码；
    2. 每个 demo 都支持 ``--fake``，用假模型把流程跑通，不消耗任何额度；
    3. demo 之间不互相 import 业务代码，只共享本模块。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

HERE = Path(__file__).resolve().parent

# Windows 控制台默认 GBK 编码，打印个别符号（如 U+2713）会直接抛
# UnicodeEncodeError 让整个 demo 崩掉。降级成 replace，保证永远只是显示成
# 问号，而不是抛异常。输出全部使用 GBK 能编码的字符。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # pragma: no cover - 某些流不支持 reconfigure
        pass


# ---------------------------------------------------------------------------
# 1. 读取 .env
# ---------------------------------------------------------------------------
def _find_env_file() -> Path | None:
    """按 脚本同级 → 上级目录 → 当前工作目录 的顺序找 .env。"""
    for candidate in (
        HERE / ".env",
        HERE.parent / ".env",
        HERE.parent.parent / ".env",
        Path.cwd() / ".env",
    ):
        if candidate.is_file():
            return candidate
    return None


ENV_FILE = _find_env_file()
if ENV_FILE is not None:
    load_dotenv(ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# 2. 模型工厂
# ---------------------------------------------------------------------------
PROVIDERS = ("deepseek", "minimax", "openai")


def _provider_config(provider: str) -> dict[str, Any]:
    """把一个 provider 归一成统一的配置字典。"""
    p = provider.strip().lower()
    if p == "deepseek":
        return {
            "key_env": "DEEPSEEK_API_KEY",
            "key": os.getenv("DEEPSEEK_API_KEY", "").strip(),
            "base_env": "DEEPSEEK_BASE_URL",
            "base": os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1",
            "model": os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-chat",
        }
    if p == "minimax":
        return {
            "key_env": "MINIMAX_API_KEY",
            "key": os.getenv("MINIMAX_API_KEY", "").strip(),
            "base_env": "MINIMAX_BASE_URL",
            "base": os.getenv("MINIMAX_BASE_URL", "").strip() or "https://api.minimax.cn/v1",
            "model": os.getenv("MINIMAX_MODEL", "").strip() or "MiniMax-M3",
        }
    if p == "openai":
        return {
            "key_env": "OPENAI_API_KEY",
            "key": os.getenv("OPENAI_API_KEY", "").strip(),
            "base_env": "OPENAI_BASE_URL",
            "base": os.getenv("OPENAI_BASE_URL", "").strip(),
            "model": os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
        }
    raise ValueError(f"未知的 LLM_PROVIDER: {provider!r}，可选：{', '.join(PROVIDERS)}")


def get_chat_model(temperature: float | None = None, **overrides):
    """按 .env 里的 LLM_PROVIDER 创建聊天模型。

    - deepseek -> langchain_deepseek.ChatDeepSeek（DeepSeek 原生端点）
    - minimax  -> langchain_openai.ChatOpenAI + base_url（MiniMax 是 OpenAI 兼容协议）
    - openai   -> langchain_openai.ChatOpenAI（也可填任何 OpenAI 兼容网关）
    """
    provider = (os.getenv("LLM_PROVIDER") or "deepseek").strip() or "deepseek"
    cfg = _provider_config(provider)

    if not cfg["key"]:
        raise RuntimeError(
            f"缺少密钥 {cfg['key_env']}。\n"
            f"  做法：把 {HERE / '.env.example'} 复制成 {HERE / '.env'}，填上真实 key。\n"
            f"  或者：给脚本加 --fake 参数，用假模型跑通流程（不消耗额度）。"
        )

    kwargs: dict[str, Any] = {"model": cfg["model"], "api_key": cfg["key"]}
    if cfg["base"]:
        kwargs["base_url"] = cfg["base"]
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs.update(overrides)

    if provider.lower() == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(**kwargs)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


def get_fake_model(responses: list[str]):
    """离线假模型：按顺序返回预设回答，用来自测流程、避免消耗额度。

    注意：FakeListChatModel 不支持工具调用（bind_tools 抛 NotImplementedError），
    所以 04/05 两个 demo 必须用真实模型。
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=list(responses))


def resolve_model(fake: bool = False, responses: list[str] | None = None, **kwargs):
    """demo 统一入口：fake=True 返回假模型，否则返回真实模型。"""
    if fake:
        return get_fake_model(responses or ["这是假模型的固定回答。"])
    return get_chat_model(**kwargs)


# ---------------------------------------------------------------------------
# 3. 小工具
# ---------------------------------------------------------------------------
class ScriptedChatModel(BaseChatModel):
    """按脚本逐条返回 AIMessage 的假模型。

    真实模型被 bind_tools 之后，会返回一条"带 tool_calls 的 AIMessage"。
    这个类就是照着那个形状造的，用来离线演示工具调用循环：

        model = ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{"name": "add", "args": {...}, "id": "call_1"}]),
            AIMessage(content="最终答案：3"),
        ])

    脚本用完后会一直重复最后一条。
    """

    script: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        """真实模型在这里把工具的 JSON Schema 塞进请求；这里只要能返回 tool_calls 即可。"""
        return self

    def reset(self) -> None:
        """把脚本指针拨回第一条，方便同一个模型重复演示不同场景。"""
        self.index = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def hr(title: str = "", width: int = 74) -> None:
    """打印一条分隔标题。"""
    print()
    print("=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def banner(no: str, title: str) -> None:
    """打印 demo 头部。"""
    hr(f"demo {no} · {title}")


def show_env_status() -> None:
    """打印当前配置状态（不泄露密钥内容）。"""
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    try:
        cfg = _provider_config(provider)
    except ValueError:
        return
    print(f"  .env 文件        : {ENV_FILE or '（未找到）'}")
    print(f"  LLM_PROVIDER     : {provider}")
    print(f"  {cfg['key_env']:<19}: {'已配置 [OK]' if cfg['key'] else '未配置 [--]'}")
    print(f"  {cfg['base_env']:<19}: {cfg['base'] or '（用官方默认）'}")
    print(f"  模型             : {cfg['model']}")


def count_tokens(text: str) -> int:
    """用 tiktoken 估算 token 数（langchain 本身就依赖 tiktoken）。"""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // 4
