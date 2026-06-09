"""协议无关的 LLM 抽象层。

定义统一消息、流式事件和 Provider 协议，让 TUI 不关心底层是 Anthropic 还是 OpenAI。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from agentcode.config import ProviderConfig


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    # 统一流式事件：text/done/err 三类信号由 TUI 按同一套逻辑消费。
    text: str = ""
    done: bool = False
    err: Exception | None = None


class Provider(Protocol):
    # 上层只依赖这个协议接口，因此新增后端时不需要改 TUI。
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]: ...


def new_provider(cfg: ProviderConfig) -> Provider:
    # 适配器延迟导入，避免未选中的 SDK 在启动时产生额外副作用。
    if cfg.protocol == "anthropic":
        from agentcode.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from agentcode.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"Unsupported protocol: {cfg.protocol}")
