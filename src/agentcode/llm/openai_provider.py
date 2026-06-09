"""OpenAI 协议适配器。

封装 AsyncOpenAI chat.completions 流式接口，并把文本增量转换为统一 StreamEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from agentcode.config import ProviderConfig
from agentcode.llm import Message, StreamEvent
from agentcode.prompt import SYSTEM_PROMPT


class OpenAIProvider:
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
        # OpenAI chat.completions 需要把 system prompt 放进 messages 第一项。
        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(_to_openai_message(message) for message in msgs)

        try:
            # SDK 已处理 SSE，适配器只负责抽取正文增量并统一成 StreamEvent。
            stream = await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                text = _extract_text_delta(chunk)
                if text:
                    yield StreamEvent(text=text)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 将 SDK 错误传回 UI 展示。
            yield StreamEvent(err=exc)


def _to_openai_message(message: Message) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def _new_client(cfg: ProviderConfig) -> AsyncOpenAI:
    # base_url 支持 OpenAI 兼容端点，例如代理或第三方兼容服务。
    if cfg.base_url:
        return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return AsyncOpenAI(api_key=cfg.api_key)


def _extract_text_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return str(getattr(delta, "content", "") or "")
