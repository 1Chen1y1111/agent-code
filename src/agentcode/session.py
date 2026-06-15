"""
AgentCode 的进程内会话封装。

负责把用户输入、对话历史和 Agent Core 生命周期事件连接起来；维护进程内
对话历史，并把上下文压缩交给专用管理器。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentcode.context import CompactionReport, ContextManager, ContextSettings
from agentcode.agent import (
    Agent,
    AgentEvent,
    AgentRunOptions,
    EventType,
    PermissionApprover,
    StopReason,
)
from agentcode.conversation import Conversation
from agentcode.llm import Message, Provider, Usage, UserMessage
from agentcode.permission import (
    PLAN_REMINDER,
    PLAN_TOOL_NAMES,
    PermissionMode,
    PermissionPolicy,
)
from agentcode.prompt import PromptBuildOptions, SupplementalInstruction
from agentcode.tool import Registry

SessionEventType = Literal["agent_start"] | EventType


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """AgentSession 对 UI/模式层暴露的统一事件。"""

    type: SessionEventType
    message: Message | None = None
    thinking: str = ""
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    args: str = ""
    result: str = ""
    is_error: bool = False
    usage: Usage | None = None
    progress: str = ""
    stop_reason: StopReason | None = None
    err: Exception | None = None
    compaction_reason: Literal["manual", "threshold", "overflow"] | None = None
    compaction: CompactionReport | None = None
    will_retry: bool = False
    error_message: str = ""

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
            usage=event.usage,
            progress=event.progress,
            stop_reason=event.stop_reason,
            err=event.err,
            compaction_reason=event.compaction_reason,
            compaction=event.compaction,
            will_retry=event.will_retry,
            error_message=event.error_message,
        )


class AgentSession:
    """进程内会话，持有 Provider、工具注册中心、提示选项和线性对话历史。"""

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        prompt_options: PromptBuildOptions | None = None,
        *,
        permission_policy: PermissionPolicy | None = None,
        permission_mode: Callable[[], PermissionMode] | None = None,
        permission_approver: PermissionApprover | None = None,
        context_settings: ContextSettings | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        """创建绑定 provider 和工具集的进程内会话。"""

        self.provider = provider
        self._registry = registry
        self._prompt_options = prompt_options or PromptBuildOptions()
        self._conversation = Conversation()
        self._permission_policy = permission_policy
        self._permission_mode = permission_mode or (lambda: "default")
        self._permission_approver = permission_approver
        self._context_manager = ContextManager(
            context_settings,
            project_root=project_root,
        )

    def messages(self) -> list[Message]:
        """返回当前会话历史副本，主要供测试和调试观察。"""

        return self._conversation.messages()

    async def prompt(self, text: str) -> AsyncIterator[SessionEvent]:
        """提交一条用户输入，并产出用户消息和 Agent Core 的完整事件流。"""

        user_message = UserMessage(content=text)
        self._conversation.add_user(text)

        yield SessionEvent(type="agent_start")
        yield SessionEvent(type="message_start", message=user_message)
        yield SessionEvent(type="message_end", message=user_message)

        mode = self._permission_mode()
        agent = Agent(self.provider, self._registry)
        run_options = AgentRunOptions(
            prompt_options=self._prompt_options,
            supplemental_instructions=_mode_supplemental_instructions(mode),
            permission_policy=self._permission_policy,
            permission_mode=mode,
            permission_approver=self._permission_approver,
            visible_tool_names=_mode_visible_tool_names(mode, self._registry),
            context_manager=self._context_manager,
        )
        async for event in agent.run(self._conversation, run_options):
            yield SessionEvent.from_agent(event)

    async def compact(
        self,
        custom_instructions: str | None = None,
    ) -> CompactionReport | None:
        """手动压缩当前 active 历史，供 `/compact` 命令调用。"""

        return await self._context_manager.compact_conversation(
            self._conversation,
            self.provider,
            self._registry.definitions(),
            custom_instructions=custom_instructions,
        )


def _mode_supplemental_instructions(
    mode: PermissionMode,
) -> tuple[SupplementalInstruction, ...]:
    """根据当前权限模式生成只在本轮生效的补充指令。"""

    if mode != "plan":
        return ()
    return (SupplementalInstruction(source="permission_mode", content=PLAN_REMINDER),)


def _mode_visible_tool_names(
    mode: PermissionMode,
    registry: Registry,
) -> tuple[str, ...] | None:
    """根据权限模式返回本轮可暴露给模型的工具名，非 plan 模式不限制。"""

    if mode != "plan":
        return None
    readonly_names = {
        name
        for name in registry.names()
        if name in PLAN_TOOL_NAMES or registry.permission_category(name) == "readonly"
    }
    return tuple(name for name in registry.names() if name in readonly_names)
