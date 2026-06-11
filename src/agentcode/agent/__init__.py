"""AgentCode 的单轮工具调用编排层。

负责把 provider 流式回复、工具执行和结果回灌串成一次完整 Agent 回合。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from agentcode.conversation import Conversation
from agentcode.llm import Provider, ToolCall, ToolResult
from agentcode.tool import DEFAULT_TIMEOUT, Registry

TOOL_ARG_PREVIEW_CHARS = 80
SINGLE_TOOL_ROUND_LIMIT_MESSAGE = "本轮已达到单轮工具调用上限，未继续执行新的工具请求。"


class Phase(Enum):
    START = "start"
    END = "end"


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """供 TUI 渲染的一次工具调用开始或结束事件。"""

    name: str
    args: str = ""
    phase: Phase = Phase.START
    result: str = ""
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    """Agent 对外事件流，调用方按字段分派渲染。"""

    thinking: str = ""
    text: str = ""
    tool: ToolEvent | None = None
    done: bool = False
    err: Exception | None = None


class Agent:
    """执行请求、工具调用、工具结果回灌和续答的单轮闭环。"""

    def __init__(self, provider: Provider, registry: Registry) -> None:
        self._provider = provider
        self._registry = registry

    async def run(self, conv: Conversation) -> AsyncIterator[Event]:
        definitions = self._registry.definitions()

        preamble = ""
        calls: list[ToolCall] = []
        async for stream_event in self._provider.stream(conv.messages(), definitions):
            if stream_event.err is not None:
                yield Event(err=stream_event.err)
                return
            if stream_event.thinking:
                yield Event(thinking=stream_event.thinking)
            if stream_event.text:
                preamble += stream_event.text
                yield Event(text=stream_event.text)
            if stream_event.tool_calls:
                calls.extend(stream_event.tool_calls)

        if not calls:
            conv.add_assistant(preamble)
            yield Event(done=True)
            return

        conv.add_assistant_with_tool_calls(preamble, calls)
        results: list[ToolResult] = []
        for call in calls:
            args_preview = _preview_args(call.input)
            yield Event(
                tool=ToolEvent(
                    name=call.name,
                    args=args_preview,
                    phase=Phase.START,
                )
            )
            result = await self._registry.execute(
                call.name, call.input, timeout=DEFAULT_TIMEOUT
            )
            yield Event(
                tool=ToolEvent(
                    name=call.name,
                    args=args_preview,
                    phase=Phase.END,
                    result=result.content,
                    is_error=result.is_error,
                )
            )
            results.append(
                ToolResult(
                    tool_call_id=call.id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

        conv.add_tool_results(results)

        final = ""
        requested_more_tools = False
        async for stream_event in self._provider.stream(conv.messages(), definitions):
            if stream_event.err is not None:
                yield Event(err=stream_event.err)
                return
            if stream_event.thinking:
                yield Event(thinking=stream_event.thinking)
            if stream_event.text:
                final += stream_event.text
                yield Event(text=stream_event.text)
            if stream_event.tool_calls:
                requested_more_tools = True

        if not final and requested_more_tools:
            final = SINGLE_TOOL_ROUND_LIMIT_MESSAGE
            yield Event(text=final)

        conv.add_assistant(final)
        yield Event(done=True)


def _preview_args(args: str) -> str:
    compact = " ".join((args or "{}").split())
    if len(compact) <= TOOL_ARG_PREVIEW_CHARS:
        return compact
    return compact[: TOOL_ARG_PREVIEW_CHARS - 3] + "..."
