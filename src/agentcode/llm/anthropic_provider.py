"""
Anthropic 协议适配器。

封装 AsyncAnthropic 的流式 messages API，并把 SDK 事件转换为统一
AssistantMessageEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

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
    ThinkingContent,
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

DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET_TOKENS = 2048
REDACTED_THINKING_TEXT = "[Reasoning redacted]"
ANTHROPIC_API = "anthropic-messages"
ANTHROPIC_PROVIDER = "anthropic"


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig, client: Any | None = None) -> None:
        """保存 Anthropic 配置，并允许测试注入假客户端。"""

        self._cfg = cfg
        self._client: Any = client or _new_client(cfg)

    @property
    def api(self) -> str:
        """返回 provider 使用的 API 协议名。"""

        return ANTHROPIC_API

    @property
    def name(self) -> str:
        """返回配置中的 provider 展示名。"""

        return self._cfg.name

    @property
    def model(self) -> str:
        """返回配置中的 Anthropic 模型名。"""

        return self._cfg.model

    @property
    def context_window(self) -> int | None:
        """返回用户配置的模型上下文窗口，缺省时交给上层使用协议默认值。"""

        return self._cfg.context_window

    async def stream(
        self, context: Context, options: StreamOptions | None = None
    ) -> AsyncIterator[AssistantMessageEvent]:
        """调用 Anthropic messages stream，并转换为统一 assistant 事件。"""

        stream_options = options or StreamOptions()
        cache_control = _cache_control(stream_options.cache_retention)
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": stream_options.max_tokens or DEFAULT_MAX_TOKENS,
            "system": _to_anthropic_system(
                context.system_prompt or SYSTEM_PROMPT,
                cache_control,
            ),
            "messages": _to_anthropic_messages(context.messages),
        }
        if stream_options.temperature is not None:
            request["temperature"] = stream_options.temperature
        if context.tools:
            request["tools"] = _to_anthropic_tools(context.tools, cache_control)
        thinking_enabled = self._cfg.thinking and not _has_tool_history(context.messages)
        thinking_request = _thinking_request(self._cfg, thinking_enabled)
        if thinking_request is not None:
            # Anthropic thinking replay 需要签名；当前协议已能保存 thinking block，
            # 但没有实现签名回放，所以已有工具历史时仍禁用 thinking。
            request["thinking"] = thinking_request

        content: list[AssistantContent] = []
        text = ""
        thinking = ""
        text_index: int | None = None
        thinking_index: int | None = None

        yield StartEvent(partial=_assistant_message(self._cfg, content))

        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    thinking_delta = _extract_thinking_delta(event) if thinking_enabled else ""
                    if thinking_delta:
                        if thinking_index is None:
                            thinking_index = len(content)
                            content.append(
                                thinking_content(
                                    "",
                                    redacted=thinking_delta == REDACTED_THINKING_TEXT,
                                )
                            )
                            yield ThinkingStartEvent(
                                content_index=thinking_index,
                                partial=_assistant_message(self._cfg, content),
                            )
                        thinking += thinking_delta
                        content[thinking_index] = thinking_content(
                            thinking,
                            redacted=thinking == REDACTED_THINKING_TEXT,
                        )
                        yield ThinkingDeltaEvent(
                            content_index=thinking_index,
                            delta=thinking_delta,
                            partial=_assistant_message(self._cfg, content),
                        )

                    text_delta = _extract_text_delta(event)
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

                final_message = await _extract_final_message(stream)
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

                calls = _extract_tool_calls(final_message)
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

                usage = _extract_usage(final_message)
                stop_reason = _extract_stop_reason(final_message)
                if stop_reason is None:
                    stop_reason = "toolUse" if calls else "stop"
                message = _assistant_message(
                    self._cfg,
                    content,
                    usage=usage,
                    stop_reason=stop_reason,
                    response_id=_string_attr(final_message, "id"),
                    response_model=_string_attr(final_message, "model"),
                )
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
        api=ANTHROPIC_API,
        provider=ANTHROPIC_PROVIDER,
        model=cfg.model,
        response_model=response_model,
        response_id=response_id,
        diagnostics=diagnostics or [],
        usage=usage or Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _done_reason(reason: ModelStopReason) -> DoneStopReason:
    """收窄 Anthropic 正常结束原因，避免 done 事件携带 error/aborted。"""

    if reason in ("stop", "length", "toolUse"):
        return reason
    return "stop"


def _thinking_request(
    cfg: ProviderConfig,
    thinking_enabled: bool,
) -> dict[str, Any] | None:
    """生成 thinking 请求参数，DeepSeek 需要显式 disabled 才是非思考。"""

    if thinking_enabled:
        return {
            "type": "enabled",
            "budget_tokens": DEFAULT_THINKING_BUDGET_TOKENS,
        }
    if _is_deepseek_endpoint(cfg):
        return {"type": "disabled"}
    return None


def _is_deepseek_endpoint(cfg: ProviderConfig) -> bool:
    """判断当前配置是否指向 DeepSeek 官方或 DeepSeek 命名模型。"""

    base_url = (cfg.base_url or "").lower()
    return "deepseek" in base_url or cfg.model.lower().startswith("deepseek-")


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """把内部消息列表转成 Anthropic messages API 的消息结构。"""

    converted: list[dict[str, Any]] = []
    pending_tool_results: list[ToolResultMessage] = []
    for message in messages:
        if isinstance(message, ToolResultMessage):
            pending_tool_results.append(message)
            continue
        if pending_tool_results:
            converted.append(_to_anthropic_tool_result_message(pending_tool_results))
            pending_tool_results = []
        converted.append(_to_anthropic_message(message))
    if pending_tool_results:
        converted.append(_to_anthropic_tool_result_message(pending_tool_results))
    return converted


def _to_anthropic_tool_result_message(
    results: list[ToolResultMessage],
) -> dict[str, Any]:
    """把连续 toolResult 消息合并成 Anthropic 要求的 user/tool_result 块。"""

    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": tool_result_text(result),
                "is_error": result.is_error,
            }
            for result in results
        ],
    }


def _to_anthropic_message(message: Message) -> dict[str, Any]:
    """把非 toolResult 的内部消息转成 Anthropic messages API 消息。"""

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": _to_anthropic_assistant_content(message.content),
        }
    return {"role": message.role, "content": _to_anthropic_user_content(message.content)}


def _to_anthropic_user_content(content: object) -> object:
    """转换 user content；字符串保持原样，block 列表转换成 Anthropic 格式。"""

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
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )
    return blocks


def _to_anthropic_assistant_content(
    content: list[AssistantContent],
) -> list[dict[str, Any]]:
    """把 assistant blocks 转成 Anthropic content，跳过无法安全 replay 的 thinking。"""

    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent) and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )
        elif isinstance(block, ThinkingContent):
            continue
    return blocks


def _to_anthropic_system(
    system_prompt: str,
    cache_control: dict[str, str] | None,
) -> str | list[dict[str, Any]]:
    """转换 system prompt；只有显式缓存请求才使用 block 形态。"""

    if cache_control is None:
        return system_prompt
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": cache_control,
        }
    ]


def _to_anthropic_tools(
    tools: list[ToolDefinition],
    cache_control: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """把统一工具定义转换成 Anthropic tool_use 可识别的格式。"""

    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        item = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        if cache_control is not None and index == len(tools) - 1:
            item["cache_control"] = cache_control
        converted.append(item)
    return converted


def _cache_control(cache_retention: str | None) -> dict[str, str] | None:
    """把统一 cache_retention 映射为 Anthropic ephemeral cache 控制。"""

    if cache_retention in (None, "none"):
        return None
    if cache_retention == "long":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _new_client(cfg: ProviderConfig) -> AsyncAnthropic:
    """创建 AsyncAnthropic 客户端，并支持兼容 base_url。"""

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
            return REDACTED_THINKING_TEXT
        return ""
    if event_type != "content_block_delta":
        return ""
    delta = getattr(event, "delta", None)
    if getattr(delta, "type", None) != "thinking_delta":
        return ""
    return str(getattr(delta, "thinking", "") or "")


async def _extract_final_message(stream: Any) -> Any | None:
    """从 Anthropic stream helper 读取最终消息；旧测试假对象可能没有该方法。"""

    get_final_message = getattr(stream, "get_final_message", None)
    if get_final_message is None:
        return None
    return await get_final_message()


def _extract_tool_calls(final_message: Any | None) -> list[ToolCall]:
    """从 Anthropic 最终消息中抽取 stop_reason=tool_use 的工具调用。"""

    if final_message is None:
        return []
    if getattr(final_message, "stop_reason", None) != "tool_use":
        return []

    calls: list[ToolCall] = []
    for block in getattr(final_message, "content", []) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        raw_arguments = getattr(block, "input", {}) or {}
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        calls.append(
            ToolCall(
                id=str(getattr(block, "id", "")),
                name=str(getattr(block, "name", "")),
                arguments=arguments,
            )
        )
    return calls


def _extract_usage(final_message: Any | None) -> Usage | None:
    """从 Anthropic 最终消息中提取统一 token 用量。"""

    usage = getattr(final_message, "usage", None)
    if usage is None:
        return None
    input_tokens = _int_attr(usage, "input_tokens")
    output_tokens = _int_attr(usage, "output_tokens")
    cache_read = _int_attr(usage, "cache_read_input_tokens", "cache_read_tokens")
    cache_write = _int_attr(
        usage,
        "cache_creation_input_tokens",
        "cache_write_input_tokens",
        "cache_write_tokens",
    )
    total_tokens = (
        _int_attr(usage, "total_tokens")
        or input_tokens
        + output_tokens
        + cache_read
        + cache_write
    )
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=total_tokens,
    )


def _extract_stop_reason(final_message: Any | None) -> ModelStopReason | None:
    """把 Anthropic stop_reason 映射成内部统一停止原因。"""

    raw = getattr(final_message, "stop_reason", None)
    if raw == "tool_use":
        return "toolUse"
    if raw == "max_tokens":
        return "length"
    if raw in {"end_turn", "stop_sequence"}:
        return "stop"
    return None


def _int_attr(obj: Any, *names: str) -> int:
    """安全读取 SDK 对象上的整数属性，兼容 None 或缺字段。"""

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


def _has_tool_history(msgs: list[Message]) -> bool:
    """判断历史中是否已有工具消息，用于决定是否禁用 thinking replay。"""

    return any(
        isinstance(message, ToolResultMessage)
        or (isinstance(message, AssistantMessage) and bool(assistant_tool_calls(message)))
        for message in msgs
    )
