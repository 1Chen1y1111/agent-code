"""
AgentCode 的进程内会话封装。

负责把用户输入、对话历史和 Agent Core 生命周期事件连接起来；当前不做落盘、
恢复、分支或压缩。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from agentcode.agent import Agent, AgentEvent, EventType
from agentcode.conversation import Conversation
from agentcode.llm import ROLE_USER, Message, Provider
from agentcode.tool import Registry

SessionEventType = Literal["agent_start", "turn_start"] | EventType


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """AgentSession 对 TUI/模式层暴露的统一事件。"""

    type: SessionEventType
    message: Message | None = None
    thinking: str = ""
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    args: str = ""
    result: str = ""
    is_error: bool = False
    err: Exception | None = None

    @classmethod
    def from_agent(cls, event: AgentEvent) -> "SessionEvent":
        """把底层 AgentEvent 原样提升为 SessionEvent。"""

        return cls(
            type=event.type,
            message=event.message,
            thinking=event.thinking,
            text=event.text,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=event.args,
            result=event.result,
            is_error=event.is_error,
            err=event.err,
        )


class AgentSession:
    """进程内会话，持有 Provider、工具注册中心和线性对话历史。"""

    def __init__(self, provider: Provider, registry: Registry) -> None:
        """创建绑定 provider 和工具集的进程内会话。"""

        self.provider = provider
        self._registry = registry
        self._conversation = Conversation()

    def messages(self) -> list[Message]:
        """返回当前会话历史副本，主要供测试和调试观察。"""

        return self._conversation.messages()

    async def prompt(self, text: str) -> AsyncIterator[SessionEvent]:
        """提交一条用户输入，并产出用户消息和 Agent Core 的完整事件流。"""

        user_message = Message(role=ROLE_USER, content=text)
        self._conversation.add_user(text)

        yield SessionEvent(type="agent_start")
        yield SessionEvent(type="turn_start")
        yield SessionEvent(type="message_start", message=user_message)
        yield SessionEvent(type="message_end", message=user_message)

        agent = Agent(self.provider, self._registry)
        async for event in agent.run(self._conversation):
            yield SessionEvent.from_agent(event)
