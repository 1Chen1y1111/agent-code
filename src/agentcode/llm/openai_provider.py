"""
OpenAI 协议适配器。

封装 AsyncOpenAI chat.completions 流式接口，并把文本、thinking 和工具调用转换为
统一 AssistantMessageEvent。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from agentcode.config import ProviderConfig
from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageDiagnostic,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    DoneStopReason,
    ErrorEvent,
    ImageContent,
    Message,
    ModelStopReason,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    assistant_tool_calls,
    diagnostic_from_exception,
    thinking_content,
    tool_result_text,
)
from agentcode.prompt import SYSTEM_PROMPT

OPENAI_COMPLETIONS_API = "openai-completions"
OPENAI_PROVIDER = "openai"


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig, client: Any | None = None) -> None:
        """保存 OpenAI 配置，并允许测试注入假客户端。"""

        self._cfg = cfg
        self._client: Any = client or _new_client(cfg)

    @property
    def api(self) -> str:
        """返回 provider 使用的 API 协议名。"""

        return OPENAI_COMPLETIONS_API

    @property
    def name(self) -> str:
        """返回配置中的 provider 展示名。"""

        return self._cfg.name

    @property
    def model(self) -> str:
        """返回配置中的 OpenAI 或兼容模型名。"""

        return self._cfg.model

    async def stream(
        self, context: Context, options: StreamOptions | None = None
    ) -> AsyncIterator[AssistantMessageEvent]:
        """调用 chat.completions 流式接口，并转换为统一 assistant 事件。"""

        stream_options = options or StreamOptions()
        messages: list[Any] = [
            {"role": "system", "content": context.system_prompt or SYSTEM_PROMPT}
        ]
        for message in context.messages:
            messages.extend(_to_openai_messages(message))
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stream_options.cache_retention not in (None, "none"):
            if stream_options.session_id:
                request["prompt_cache_key"] = stream_options.session_id
            if stream_options.cache_retention == "long":
                request["prompt_cache_retention"] = "24h"
        if stream_options.temperature is not None:
            request["temperature"] = stream_options.temperature
        if stream_options.max_tokens is not None:
            request["max_tokens"] = stream_options.max_tokens
        if context.tools:
            request["tools"] = _to_openai_tools(context.tools)

        content: list[AssistantContent] = []
        text = ""
        thinking = ""
        text_index: int | None = None
        thinking_index: int | None = None
        tool_calls_buf: dict[int, dict[str, str]] = {}
        usage: Usage | None = None
        stop_reason: ModelStopReason | None = None
        response_id: str | None = None
        response_model: str | None = None

        yield StartEvent(partial=_assistant_message(self._cfg, content))

        try:
            stream = await self._client.chat.completions.create(**request)
            async for chunk in stream:
                response_id = _string_attr(chunk, "id") or response_id
                response_model = _string_attr(chunk, "model") or response_model
                chunk_usage = _extract_usage(chunk)
                if chunk_usage is not None:
                    usage = chunk_usage
                chunk_stop_reason = _extract_stop_reason(chunk)
                if chunk_stop_reason is not None:
                    stop_reason = chunk_stop_reason
                _merge_tool_call_deltas(chunk, tool_calls_buf)

                thinking_delta = _extract_thinking_delta(chunk)
                if thinking_delta:
                    if thinking_index is None:
                        thinking_index = len(content)
                        content.append(thinking_content(""))
                        yield ThinkingStartEvent(
                            content_index=thinking_index,
                            partial=_assistant_message(self._cfg, content),
                        )
                    thinking += thinking_delta
                    content[thinking_index] = thinking_content(thinking)
                    yield ThinkingDeltaEvent(
                        content_index=thinking_index,
                        delta=thinking_delta,
                        partial=_assistant_message(self._cfg, content),
                    )

                text_delta = _extract_text_delta(chunk)
                if text_delta:
                    if text_index is None:
                        text_index = len(content)
                        content.append(TextContent(text=""))
                        yield TextStartEvent(
                            content_index=text_index,
                            partial=_assistant_message(self._cfg, content),
                        )
                    text += text_delta
                    content[text_index] = TextContent(text=text)
                    yield TextDeltaEvent(
                        content_index=text_index,
                        delta=text_delta,
                        partial=_assistant_message(self._cfg, content),
                    )

            if thinking_index is not None:
                yield ThinkingEndEvent(
                    content_index=thinking_index,
                    content=thinking,
                    partial=_assistant_message(self._cfg, content),
                )
            if text_index is not None:
                yield TextEndEvent(
                    content_index=text_index,
                    content=text,
                    partial=_assistant_message(self._cfg, content),
                )

            calls = _build_tool_calls(tool_calls_buf)
            for call in calls:
                content_index = len(content)
                content.append(call)
                partial = _assistant_message(self._cfg, content)
                yield ToolCallStartEvent(
                    content_index=content_index,
                    partial=partial,
                )
                yield ToolCallEndEvent(
                    content_index=content_index,
                    tool_call=call,
                    partial=partial,
                )

            if calls and stop_reason is None:
                stop_reason = "toolUse"
            stop_reason = stop_reason or "stop"
            message = _assistant_message(
                self._cfg,
                content,
                usage=usage,
                stop_reason=stop_reason,
                response_id=response_id,
                response_model=response_model,
            )
            if stop_reason in ("error", "aborted"):
                yield ErrorEvent(reason=stop_reason, error=message)
            else:
                yield DoneEvent(reason=_done_reason(stop_reason), message=message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 将 SDK 错误传回 UI 展示。
            error = _assistant_message(
                self._cfg,
                content,
                stop_reason="error",
                error_message=str(exc),
                diagnostics=[
                    diagnostic_from_exception("provider_stream_error", exc)
                ],
            )
            yield ErrorEvent(
                reason="error",
                error=error,
                err=exc,
            )


def _assistant_message(
    cfg: ProviderConfig,
    content: list[AssistantContent],
    usage: Usage | None = None,
    stop_reason: ModelStopReason = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    response_model: str | None = None,
    diagnostics: list[AssistantMessageDiagnostic] | None = None,
) -> AssistantMessage:
    """用 provider 配置补齐统一 AssistantMessage 元数据。"""

    return AssistantMessage(
        content=list(content),
        api=OPENAI_COMPLETIONS_API,
        provider=OPENAI_PROVIDER,
        model=cfg.model,
        response_model=response_model,
        response_id=response_id,
        diagnostics=diagnostics or [],
        usage=usage or Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _done_reason(reason: ModelStopReason) -> DoneStopReason:
    """收窄 OpenAI 正常结束原因，避免 done 事件携带 error/aborted。"""

    if reason in ("stop", "length", "toolUse"):
        return reason
    return "stop"


def _to_openai_messages(message: Message) -> list[dict[str, Any]]:
    """把内部 Message 转成 OpenAI chat.completions 消息片段。"""

    if isinstance(message, ToolResultMessage):
        return [
            {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": tool_result_text(message),
            }
        ]
    if isinstance(message, AssistantMessage):
        text = _assistant_openai_text(message)
        calls = assistant_tool_calls(message)
        if calls:
            return [
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": _json_arguments(call.arguments),
                            },
                        }
                        for call in calls
                    ],
                }
            ]
        return [{"role": "assistant", "content": text}]
    return [{"role": message.role, "content": _to_openai_user_content(message.content)}]


def _assistant_openai_text(message: AssistantMessage) -> str:
    """提取可 replay 给 OpenAI 的 assistant 文本，thinking block 不回放。"""

    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


def _to_openai_user_content(content: object) -> object:
    """转换 user content；字符串保持原样，block 列表转成 OpenAI 多模态格式。"""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.mime_type};base64,{block.data}",
                    },
                }
            )
    return blocks


def _to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """把统一工具定义转换成 OpenAI function tool 格式。"""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _new_client(cfg: ProviderConfig) -> AsyncOpenAI:
    """创建 AsyncOpenAI 客户端，并支持 OpenAI 兼容 base_url。"""

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


def _extract_usage(chunk: Any) -> Usage | None:
    """从 OpenAI 流式 chunk 中提取统一 token 用量。"""

    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    input_tokens = _int_attr(usage, "prompt_tokens", "input_tokens")
    output_tokens = _int_attr(usage, "completion_tokens", "output_tokens")
    prompt_details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    cache_read = _int_attr(
        prompt_details,
        "cached_tokens",
    ) or _int_attr(usage, "prompt_cache_hit_tokens")
    cache_write = _int_attr(
        prompt_details,
        "cache_write_tokens",
        "cache_creation_input_tokens",
    )
    uncached_input = max(0, input_tokens - cache_read - cache_write)
    total_tokens = (
        _int_attr(usage, "total_tokens")
        or uncached_input + output_tokens + cache_read + cache_write
    )
    return Usage(
        input=uncached_input,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=total_tokens,
    )


def _extract_stop_reason(chunk: Any) -> ModelStopReason | None:
    """从 OpenAI chunk 的 finish_reason 映射出内部停止原因。"""

    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    raw = getattr(choices[0], "finish_reason", None)
    if raw == "tool_calls":
        return "toolUse"
    if raw == "length":
        return "length"
    if raw == "stop":
        return "stop"
    if raw == "content_filter":
        return "error"
    return None


def _int_attr(obj: Any, *names: str) -> int:
    """按多个候选字段安全读取整数属性，缺失或 None 时返回 0。"""

    if obj is None:
        return 0
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return 0


def _string_attr(obj: Any | None, name: str) -> str | None:
    """安全读取 SDK 对象上的字符串属性，空值统一视为不存在。"""

    value = getattr(obj, name, None)
    if value is None:
        return None
    text = str(value)
    return text or None


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
    """在流结束后把工具调用缓冲区转换为统一 ToolCall block。"""

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
                arguments=_json_object(item.get("args") or "{}"),
            )
        )
    return calls


def _json_arguments(arguments: dict[str, Any]) -> str:
    """把 ToolCall.arguments 稳定序列化为 OpenAI 所需 JSON 字符串。"""

    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _json_object(raw: str) -> dict[str, Any]:
    """把 OpenAI function arguments 字符串解析成统一对象参数。"""

    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
