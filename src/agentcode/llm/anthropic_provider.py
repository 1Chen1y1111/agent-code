"""
Anthropic 协议适配器。

封装 AsyncAnthropic 的流式 messages API，并把 SDK 事件转换为统一 StreamEvent。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

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

DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET_TOKENS = 2048


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig, client: Any | None = None) -> None:
        """保存 Anthropic 配置，并允许测试注入假客户端。"""

        self._cfg = cfg
        self._client: Any = client or _new_client(cfg)

    @property
    def name(self) -> str:
        """返回配置中的 provider 展示名。"""

        return self._cfg.name

    @property
    def model(self) -> str:
        """返回配置中的 Anthropic 模型名。"""

        return self._cfg.model

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """调用 Anthropic messages stream，并统一文本、thinking 和工具调用事件。"""

        # system prompt 由适配器注入，Conversation 只保存 user/assistant 历史。
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [_to_anthropic_message(message) for message in msgs],
        }
        if tools:
            request["tools"] = _to_anthropic_tools(tools)
        if self._cfg.thinking and not _has_tool_history(msgs):
            # 扩展思考只在 Anthropic 协议开启；tool history 存在时 Anthropic
            # 对 thinking replay 有签名要求，当前轻量消息模型先禁用。
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": DEFAULT_THINKING_BUDGET_TOKENS,
            }

        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    # Anthropic SDK 会同时产生 raw delta 和 helper text 事件；只吃 text，
                    # 否则同一个 token 会被追加两次。
                    thinking = _extract_thinking_delta(event)
                    if thinking:
                        yield StreamEvent(thinking=thinking)
                    text = _extract_text_delta(event)
                    if text:
                        yield StreamEvent(text=text)
                calls = await _extract_tool_calls(stream)
                if calls:
                    yield StreamEvent(tool_calls=calls)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 将 SDK 错误传回 UI 展示。
            yield StreamEvent(err=exc)


def _to_anthropic_message(message: Message) -> dict[str, Any]:
    """把内部 Message 转成 Anthropic messages API 的消息结构。"""

    if message.role == ROLE_TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in message.tool_results
            ],
        }
    if message.role == ROLE_ASSISTANT and message.tool_calls:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        content.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": _json_object(call.input),
            }
            for call in message.tool_calls
        )
        return {"role": message.role, "content": content}
    return {"role": message.role, "content": message.content}


def _to_anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """把统一工具定义转换成 Anthropic tool_use 可识别的格式。"""

    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def _new_client(cfg: ProviderConfig) -> AsyncAnthropic:
    """创建 AsyncAnthropic 客户端，并支持兼容 base_url。"""

    # base_url 为空时走官方默认端点；非空时接兼容服务。
    if cfg.base_url:
        return AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url)
    return AsyncAnthropic(api_key=cfg.api_key)


def _extract_text_delta(event: Any) -> str:
    """只从 Anthropic helper text 事件提取可见文本，避免重复 token。"""

    if getattr(event, "type", None) != "text":
        return ""
    return str(getattr(event, "text", "") or "")


def _extract_thinking_delta(event: Any) -> str:
    """从 Anthropic content block delta 中提取 extended thinking 内容。"""

    event_type = getattr(event, "type", None)
    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) == "redacted_thinking":
            return "[Reasoning redacted]"
        return ""
    if event_type != "content_block_delta":
        return ""
    delta = getattr(event, "delta", None)
    if getattr(delta, "type", None) != "thinking_delta":
        return ""
    return str(getattr(delta, "thinking", "") or "")


async def _extract_tool_calls(stream: Any) -> list[ToolCall]:
    """读取 Anthropic 最终消息，并抽取 stop_reason=tool_use 的工具调用。"""

    get_final_message = getattr(stream, "get_final_message", None)
    if get_final_message is None:
        return []
    final_message = await get_final_message()
    if getattr(final_message, "stop_reason", None) != "tool_use":
        return []

    calls: list[ToolCall] = []
    for block in getattr(final_message, "content", []) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        calls.append(
            ToolCall(
                id=str(getattr(block, "id", "")),
                name=str(getattr(block, "name", "")),
                input=json.dumps(getattr(block, "input", {}) or {}),
            )
        )
    return calls


def _json_object(raw: str) -> dict[str, Any]:
    """把工具参数字符串解析为对象，解析失败时给 Anthropic 一个空对象。"""

    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _has_tool_history(msgs: list[Message]) -> bool:
    """判断历史中是否已有工具消息，用于决定是否禁用 thinking replay。"""

    return any(message.tool_calls or message.tool_results for message in msgs)
