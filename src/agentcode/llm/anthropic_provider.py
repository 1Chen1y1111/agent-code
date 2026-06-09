"""Anthropic 协议适配器。

封装 AsyncAnthropic 的流式 messages API，并把 SDK 事件转换为统一 StreamEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from agentcode.config import ProviderConfig
from agentcode.llm import Message, StreamEvent
from agentcode.prompt import SYSTEM_PROMPT


DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET_TOKENS = 2048


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig, client: Any | None = None) -> None:
        self._cfg = cfg
        self._client: Any = client or _new_client(cfg)

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model(self) -> str:
        return self._cfg.model

    async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
        # system prompt 由适配器注入，Conversation 只保存 user/assistant 历史。
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [_to_anthropic_message(message) for message in msgs],
        }
        if self._cfg.thinking:
            # 扩展思考只在 Anthropic 协议开启；thinking 增量不会向 UI 暴露。
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": DEFAULT_THINKING_BUDGET_TOKENS,
            }

        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    # Anthropic SDK 会同时产生 raw delta 和 helper text 事件；只吃 text，
                    # 否则同一个 token 会被追加两次。
                    text = _extract_text_delta(event)
                    if text:
                        yield StreamEvent(text=text)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 将 SDK 错误传回 UI 展示。
            yield StreamEvent(err=exc)


def _to_anthropic_message(message: Message) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def _new_client(cfg: ProviderConfig) -> AsyncAnthropic:
    # base_url 为空时走官方默认端点；非空时接兼容服务。
    if cfg.base_url:
        return AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url)
    return AsyncAnthropic(api_key=cfg.api_key)


def _extract_text_delta(event: Any) -> str:
    if getattr(event, "type", None) != "text":
        return ""
    return str(getattr(event, "text", "") or "")
