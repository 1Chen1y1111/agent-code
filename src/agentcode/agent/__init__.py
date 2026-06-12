"""
AgentCode 的底层 Agent Core。

负责单轮模型请求、工具执行、工具结果回灌和生命周期事件输出；不处理 TUI、
配置加载、会话持久化或用户输入命令。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from agentcode.conversation import Conversation
from agentcode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    Message,
    Provider,
    ToolCall,
    ToolResult,
)
from agentcode.tool import DEFAULT_TIMEOUT, Registry

TOOL_ARG_PREVIEW_CHARS = 80
SINGLE_TOOL_ROUND_LIMIT_MESSAGE = "本轮已达到单轮工具调用上限，未继续执行新的工具请求。"

EventType = Literal[
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "turn_end",
    "agent_end",
    "error",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Agent Core 对外输出的 pi 风格生命周期事件。"""

    type: EventType
    message: Message | None = None
    thinking: str = ""
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    args: str = ""
    result: str = ""
    is_error: bool = False
    err: Exception | None = None


class Agent:
    """执行一次用户回合中的模型请求和单轮工具调用。"""

    def __init__(self, provider: Provider, registry: Registry) -> None:
        """绑定本回合要使用的模型 provider 和工具注册中心。"""

        self._provider = provider
        self._registry = registry

    async def run(self, conv: Conversation) -> AsyncIterator[AgentEvent]:
        """从已有对话历史继续执行当前回合。

        当前阶段只允许一次工具批次：模型拿到工具结果后可以续答，但如果再次请求
        工具，会返回固定提示并停止，自动多轮 Agent Loop 留给下一阶段。
        """

        definitions = self._registry.definitions()

        yield AgentEvent(
            type="message_start",
            message=Message(role=ROLE_ASSISTANT),
        )
        preamble = ""
        calls: list[ToolCall] = []
        async for stream_event in self._provider.stream(conv.messages(), definitions):
            if stream_event.err is not None:
                yield AgentEvent(type="error", err=stream_event.err)
                return
            if stream_event.thinking:
                yield AgentEvent(
                    type="message_update",
                    thinking=stream_event.thinking,
                )
            if stream_event.text:
                preamble += stream_event.text
                yield AgentEvent(type="message_update", text=stream_event.text)
            if stream_event.tool_calls:
                calls.extend(stream_event.tool_calls)

        if not calls:
            message = Message(role=ROLE_ASSISTANT, content=preamble)
            conv.add_assistant(preamble)
            yield AgentEvent(type="message_end", message=message)
            yield AgentEvent(type="turn_end", message=message)
            yield AgentEvent(type="agent_end")
            return

        assistant_message = Message(
            role=ROLE_ASSISTANT,
            content=preamble,
            tool_calls=list(calls),
        )
        conv.add_assistant_with_tool_calls(preamble, calls)
        yield AgentEvent(type="message_end", message=assistant_message)

        results: list[ToolResult] = []
        for call in calls:
            args_preview = _preview_args(call.input)
            yield AgentEvent(
                type="tool_execution_start",
                tool_call_id=call.id,
                tool_name=call.name,
                args=args_preview,
            )
            result = await self._registry.execute(
                call.name, call.input, timeout=DEFAULT_TIMEOUT
            )
            yield AgentEvent(
                type="tool_execution_end",
                tool_call_id=call.id,
                tool_name=call.name,
                args=args_preview,
                result=result.content,
                is_error=result.is_error,
            )
            results.append(
                ToolResult(
                    tool_call_id=call.id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

        tool_message = Message(role=ROLE_TOOL, tool_results=list(results))
        conv.add_tool_results(results)
        yield AgentEvent(type="message_start", message=tool_message)
        yield AgentEvent(type="message_end", message=tool_message)

        yield AgentEvent(
            type="message_start",
            message=Message(role=ROLE_ASSISTANT),
        )
        final = ""
        requested_more_tools = False
        async for stream_event in self._provider.stream(conv.messages(), definitions):
            if stream_event.err is not None:
                yield AgentEvent(type="error", err=stream_event.err)
                return
            if stream_event.thinking:
                yield AgentEvent(
                    type="message_update",
                    thinking=stream_event.thinking,
                )
            if stream_event.text:
                final += stream_event.text
                yield AgentEvent(type="message_update", text=stream_event.text)
            if stream_event.tool_calls:
                requested_more_tools = True

        if not final and requested_more_tools:
            final = SINGLE_TOOL_ROUND_LIMIT_MESSAGE
            yield AgentEvent(type="message_update", text=final)

        final_message = Message(role=ROLE_ASSISTANT, content=final)
        conv.add_assistant(final)
        yield AgentEvent(type="message_end", message=final_message)
        yield AgentEvent(type="turn_end", message=final_message)
        yield AgentEvent(type="agent_end")


def _preview_args(args: str) -> str:
    """生成工具参数的单行预览，避免 TUI 中长 JSON 撑开消息。"""

    compact = " ".join((args or "{}").split())
    if len(compact) <= TOOL_ARG_PREVIEW_CHARS:
        return compact
    return compact[: TOOL_ARG_PREVIEW_CHARS - 3] + "..."
