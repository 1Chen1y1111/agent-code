"""
OpenAI 协议适配器。

封装 AsyncOpenAI chat.completions 流式接口，并把文本增量转换为统一 StreamEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from agentcode.config import ProviderConfig
from agentcode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from agentcode.prompt import SYSTEM_PROMPT


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig, client: Any | None = None) -> None:
        """保存 OpenAI 配置，并允许测试注入假客户端。"""

        self._cfg = cfg
        self._client: Any = client or _new_client(cfg)

    @property
    def name(self) -> str:
        """返回配置中的 provider 展示名。"""

        return self._cfg.name

    @property
    def model(self) -> str:
        """返回配置中的 OpenAI 或兼容模型名。"""

        return self._cfg.model

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """调用 chat.completions 流式接口，并统一文本、thinking 和工具调用事件。"""

        # OpenAI chat.completions 需要把 system prompt 放进 messages 第一项。
        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in msgs:
            messages.extend(_to_openai_messages(message))
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            request["tools"] = _to_openai_tools(tools)

        try:
            # SDK 已处理 SSE，适配器只负责抽取正文增量并统一成 StreamEvent。
            stream = await self._client.chat.completions.create(**request)
            tool_calls_buf: dict[int, dict[str, str]] = {}
            async for chunk in stream:
                _merge_tool_call_deltas(chunk, tool_calls_buf)
                thinking = _extract_thinking_delta(chunk)
                if thinking:
                    yield StreamEvent(thinking=thinking)
                text = _extract_text_delta(chunk)
                if text:
                    yield StreamEvent(text=text)
            calls = _build_tool_calls(tool_calls_buf)
            if calls:
                yield StreamEvent(tool_calls=calls)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 将 SDK 错误传回 UI 展示。
            yield StreamEvent(err=exc)


def _to_openai_messages(message: Message) -> list[dict[str, Any]]:
    """把内部 Message 转成 OpenAI chat.completions 消息片段。"""

    if message.role == ROLE_TOOL:
        return [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            }
            for result in message.tool_results
        ]
    if message.role == ROLE_ASSISTANT and message.tool_calls:
        return [
            {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.input or "{}",
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        ]
    return [{"role": message.role, "content": message.content}]


def _to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """把统一工具定义转换成 OpenAI function tool 格式。"""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _new_client(cfg: ProviderConfig) -> AsyncOpenAI:
    """创建 AsyncOpenAI 客户端，并支持 OpenAI 兼容 base_url。"""

    # base_url 支持 OpenAI 兼容端点，例如代理或第三方兼容服务。
    if cfg.base_url:
        return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return AsyncOpenAI(api_key=cfg.api_key)


def _extract_text_delta(chunk: Any) -> str:
    """从一个 OpenAI 流式 chunk 中提取可见文本增量。"""

    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return str(getattr(delta, "content", "") or "")


def _extract_thinking_delta(chunk: Any) -> str:
    """兼容不同 OpenAI-like 字段名，提取 reasoning/thinking 增量。"""

    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    for field in ("reasoning_content", "reasoning", "reasoning_text"):
        value = getattr(delta, field, None)
        if value:
            return str(value)
    return ""


def _merge_tool_call_deltas(
    chunk: Any, tool_calls_buf: dict[int, dict[str, str]]
) -> None:
    """把 OpenAI 分片返回的工具名和 JSON 参数碎片合并到缓冲区。"""

    choices = getattr(chunk, "choices", None)
    if not choices:
        return
    delta = getattr(choices[0], "delta", None)
    for fallback_index, tool_call in enumerate(getattr(delta, "tool_calls", []) or []):
        index = getattr(tool_call, "index", fallback_index)
        buf = tool_calls_buf.setdefault(index, {})
        tool_call_id = getattr(tool_call, "id", None)
        if tool_call_id:
            buf["id"] = str(tool_call_id)
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        name = getattr(function, "name", None)
        if name:
            buf["name"] = str(name)
        arguments = getattr(function, "arguments", None)
        if arguments:
            buf["args"] = buf.get("args", "") + str(arguments)


def _build_tool_calls(tool_calls_buf: dict[int, dict[str, str]]) -> list[ToolCall]:
    """在流结束后把工具调用缓冲区转换为协议无关 ToolCall。"""

    calls: list[ToolCall] = []
    for index in sorted(tool_calls_buf):
        item = tool_calls_buf[index]
        name = item.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=item.get("id", f"call_{index}"),
                name=name,
                input=item.get("args") or "{}",
            )
        )
    return calls
