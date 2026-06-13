"""
AgentCode 的底层 Agent Core。

负责 ReAct 循环、模型流式收集、工具分批执行和生命周期事件输出；不处理
UI、配置加载、会话持久化或用户输入命令。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Literal

from agentcode.conversation import Conversation
from agentcode.llm import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Provider,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    StreamOptions,
    assistant_tool_calls,
    text_content,
)
from agentcode.prompt import (
    DEFAULT_TOOL_SNIPPETS,
    PromptBuildOptions,
    SupplementalInstruction,
    build_system_prompt,
    format_supplemental_instruction,
)
from agentcode.permission import (
    PermissionApproval,
    PermissionCheck,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PermissionSource,
    denied_tool_result_text,
)
from agentcode.tool import (
    DEFAULT_TIMEOUT,
    ExecutionMode,
    Registry,
    ToolResult,
    content_text,
    text_result,
)

TOOL_ARG_PREVIEW_CHARS = 80
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_UNKNOWN_TOOLS = 2
MAX_ITERATIONS_MESSAGE = "已达到本轮 Agent Loop 迭代上限，已停止继续调用工具。"
UNKNOWN_TOOL_LIMIT_MESSAGE = "连续请求未知工具，Agent Loop 已停止。"
PermissionApprover = Callable[[PermissionRequest], Awaitable[PermissionApproval]]

EventType = Literal[
    "turn_start",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "turn_end",
    "agent_end",
    "progress",
    "usage",
    "error",
]
StopReason = Literal[
    "completed",
    "max_iterations",
    "unknown_tool_limit",
    "error",
    "cancelled",
    "tool_terminated",
]


@dataclass(frozen=True, slots=True)
class AgentRunOptions:
    """控制一次 Agent Loop 的安全边界。"""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_unknown_tools: int = DEFAULT_MAX_UNKNOWN_TOOLS
    prompt_options: PromptBuildOptions = field(default_factory=PromptBuildOptions)
    supplemental_instructions: tuple[SupplementalInstruction, ...] = ()
    cache_retention: Literal["none", "short", "long"] | None = "short"
    session_id: str | None = None
    permission_policy: PermissionPolicy | None = None
    permission_mode: PermissionMode = "default"
    permission_approver: PermissionApprover | None = None
    visible_tool_names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Agent Core 对外输出的统一生命周期事件。"""

    type: EventType
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


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """一个工具调用执行完成后需要回灌给模型的结果。"""

    call: ToolCall
    result: ToolResult
    args_preview: str


class Agent:
    """执行一次用户回合中的 ReAct Agent Loop。"""

    def __init__(self, provider: Provider, registry: Registry) -> None:
        """绑定本回合要使用的模型 provider 和工具注册中心。"""

        self._provider = provider
        self._registry = registry

    async def run(
        self,
        conv: Conversation,
        options: AgentRunOptions | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """从已有对话历史继续执行，直到模型停止请求工具或触发安全边界。"""

        run_options = options or AgentRunOptions()
        unknown_tool_turns = 0
        iteration = 0

        try:
            while True:
                if iteration >= run_options.max_iterations:
                    for event in _append_stop_message(
                        conv, MAX_ITERATIONS_MESSAGE, "max_iterations"
                    ):
                        yield event
                    return

                iteration += 1
                yield AgentEvent(
                    type="turn_start",
                    progress=f"开始第 {iteration} 轮 Agent Loop",
                )
                yield AgentEvent(
                    type="progress",
                    progress=f"LLM streaming iteration {iteration}",
                )
                assistant_message: AssistantMessage | None = None
                message_started = False
                tools = _visible_tool_definitions(
                    self._registry.definitions(),
                    run_options.visible_tool_names,
                )
                context = _build_context(
                    conv,
                    tools,
                    run_options.prompt_options,
                    run_options.supplemental_instructions,
                )
                stream_options = StreamOptions(
                    cache_retention=run_options.cache_retention,
                    session_id=run_options.session_id,
                )
                async for stream_event in self._provider.stream(
                    context, stream_options
                ):
                    if stream_event.type == "start":
                        message_started = True
                        yield AgentEvent(
                            type="message_start",
                            message=stream_event.partial or AssistantMessage(),
                        )
                        continue
                    if isinstance(stream_event, ErrorEvent):
                        yield AgentEvent(
                            type="error",
                            err=stream_event.err,
                            stop_reason="error",
                        )
                        yield AgentEvent(type="agent_end", stop_reason="error")
                        return
                    if isinstance(stream_event, ThinkingDeltaEvent) and stream_event.delta:
                        yield AgentEvent(
                            type="message_update",
                            thinking=stream_event.delta,
                        )
                    if isinstance(stream_event, TextDeltaEvent) and stream_event.delta:
                        yield AgentEvent(type="message_update", text=stream_event.delta)
                    if isinstance(stream_event, DoneEvent):
                        assistant_message = stream_event.message

                if not message_started:
                    yield AgentEvent(type="message_start", message=AssistantMessage())
                if assistant_message is None:
                    assistant_message = AssistantMessage(
                        content=[text_content("模型没有返回有效响应。")],
                        stop_reason="error",
                        error_message="missing done event",
                    )
                yield AgentEvent(type="usage", usage=assistant_message.usage)
                conv.add_assistant_message(assistant_message)
                yield AgentEvent(type="message_end", message=assistant_message)

                tool_calls = assistant_tool_calls(assistant_message)
                if not tool_calls:
                    yield AgentEvent(type="turn_end", message=assistant_message)
                    yield AgentEvent(type="agent_end", stop_reason="completed")
                    return

                unavailable = [
                    call
                    for call in tool_calls
                    if not _tool_is_available(self._registry, call.name)
                ]
                unknown_tool_turns = unknown_tool_turns + 1 if unavailable else 0

                outcomes: list[_ToolOutcome] = []
                async for tool_event, outcome in _execute_tool_calls(
                    self._registry,
                    tool_calls,
                    run_options,
                ):
                    if tool_event is not None:
                        yield tool_event
                    if outcome is not None:
                        outcomes.append(outcome)

                tool_results = [
                    ToolResultMessage(
                        tool_call_id=outcome.call.id,
                        tool_name=outcome.call.name,
                        content=outcome.result.content,
                        details=outcome.result.details,
                        is_error=outcome.result.is_error,
                    )
                    for outcome in outcomes
                ]
                conv.add_tool_results(tool_results)
                for tool_message in tool_results:
                    yield AgentEvent(type="message_start", message=tool_message)
                    yield AgentEvent(type="message_end", message=tool_message)
                yield AgentEvent(type="turn_end", message=assistant_message)

                if outcomes and all(outcome.result.terminate for outcome in outcomes):
                    yield AgentEvent(type="agent_end", stop_reason="tool_terminated")
                    return

                if unknown_tool_turns >= run_options.max_unknown_tools:
                    for event in _append_stop_message(
                        conv,
                        UNKNOWN_TOOL_LIMIT_MESSAGE,
                        "unknown_tool_limit",
                    ):
                        yield event
                    return
        except asyncio.CancelledError:
            yield AgentEvent(type="agent_end", stop_reason="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - Core 兜底转换为事件，避免 UI 卡在 streaming。
            yield AgentEvent(type="error", err=exc, stop_reason="error")
            yield AgentEvent(type="agent_end", stop_reason="error")


async def _execute_tool_calls(
    registry: Registry,
    calls: list[ToolCall],
    run_options: AgentRunOptions,
) -> AsyncIterator[tuple[AgentEvent | None, _ToolOutcome | None]]:
    """按工具安全策略分批执行，并保持结果回灌顺序稳定。"""

    indexed_outcomes: list[tuple[int, _ToolOutcome]] = []
    for batch in _tool_batches(registry, calls):
        if _batch_execution_mode(registry, batch) == "parallel":
            for _, call in batch:
                yield _tool_start_event(call), None
            results = await asyncio.gather(
                *[
                    _run_tool(registry, index, call, run_options)
                    for index, call in batch
                ]
            )
            for index, outcome, updates in results:
                for update in updates:
                    yield _tool_update_event(outcome.call, update), None
                indexed_outcomes.append((index, outcome))
                yield _tool_end_event(outcome), None
            continue

        for index, call in batch:
            yield _tool_start_event(call), None
            _, outcome, updates = await _run_tool(registry, index, call, run_options)
            for update in updates:
                yield _tool_update_event(outcome.call, update), None
            indexed_outcomes.append((index, outcome))
            yield _tool_end_event(outcome), None

    for _, outcome in sorted(indexed_outcomes, key=lambda item: item[0]):
        yield None, outcome


def _tool_batches(
    registry: Registry,
    calls: list[ToolCall],
) -> list[list[tuple[int, ToolCall]]]:
    """把连续 parallel 工具合并为并发批，sequential 工具保持单独串行批。"""

    batches: list[list[tuple[int, ToolCall]]] = []
    current_parallel: list[tuple[int, ToolCall]] = []
    for index, call in enumerate(calls):
        if _call_execution_mode(registry, call) == "parallel":
            current_parallel.append((index, call))
            continue
        if current_parallel:
            batches.append(current_parallel)
            current_parallel = []
        batches.append([(index, call)])
    if current_parallel:
        batches.append(current_parallel)
    return batches


def _batch_execution_mode(
    registry: Registry,
    batch: list[tuple[int, ToolCall]],
) -> ExecutionMode:
    """返回批次执行模式；只有全部只读时才允许并发。"""

    if all(_call_execution_mode(registry, call) == "parallel" for _, call in batch):
        return "parallel"
    return "sequential"


def _call_execution_mode(
    registry: Registry,
    call: ToolCall,
) -> ExecutionMode:
    """返回单个工具调用的执行策略，未知工具按串行错误处理。"""

    if not _tool_is_available(registry, call.name):
        return "sequential"
    return registry.execution_mode(call.name) or "sequential"


async def _run_tool(
    registry: Registry,
    index: int,
    call: ToolCall,
    run_options: AgentRunOptions,
) -> tuple[int, _ToolOutcome, list[ToolResult]]:
    """执行单个工具调用，并把未知工具转成可回灌的错误结果。"""

    updates: list[ToolResult] = []
    if not _tool_is_available(registry, call.name):
        result = text_result(f"未知工具: {call.name}", is_error=True)
    else:
        result = await _run_permitted_tool(registry, call, run_options, updates)
    return (
        index,
        _ToolOutcome(
            call=call,
            result=result,
            args_preview=_preview_args(call.arguments),
        ),
        updates,
    )


async def _run_permitted_tool(
    registry: Registry,
    call: ToolCall,
    run_options: AgentRunOptions,
    updates: list[ToolResult],
) -> ToolResult:
    """先做权限判定，通过后再执行真实工具。"""

    if run_options.permission_policy is not None:
        check = run_options.permission_policy.evaluate(
            call.name,
            call.arguments,
            run_options.permission_mode,
        )
        if check.verdict == "deny":
            return _permission_denied_result(check.source, check.reason)
        if check.verdict == "ask":
            approval = await _resolve_permission_request(check, run_options)
            if approval == "cancel":
                return _permission_denied_result(
                    "human",
                    "用户取消权限确认，当前轮停止",
                    terminate=True,
                )
            if approval == "deny_once":
                return _permission_denied_result("human", "用户拒绝本次工具调用")
            if approval == "allow_always" and check.request is not None:
                run_options.permission_policy.remember_allow(check.request)

    return await registry.execute(
        call.id,
        call.name,
        call.arguments,
        timeout=DEFAULT_TIMEOUT,
        on_update=updates.append,
    )


async def _resolve_permission_request(
    check: PermissionCheck,
    run_options: AgentRunOptions,
) -> PermissionApproval:
    """把 Ask 判定交给上层审批回调，缺失回调时保守拒绝。"""

    if check.request is None or run_options.permission_approver is None:
        return "deny_once"
    return await run_options.permission_approver(check.request)


def _permission_denied_result(
    source: PermissionSource,
    reason: str,
    *,
    terminate: bool = False,
) -> ToolResult:
    """创建结构化权限拒绝工具结果，供模型继续调整策略。"""

    return text_result(
        denied_tool_result_text(source, reason),
        is_error=True,
        details={
            "permission_denied": True,
            "source": source,
            "reason": reason,
        },
        terminate=terminate,
    )


def _tool_start_event(call: ToolCall) -> AgentEvent:
    """生成工具执行开始事件。"""

    return AgentEvent(
        type="tool_execution_start",
        tool_call_id=call.id,
        tool_name=call.name,
        args=_preview_args(call.arguments),
    )


def _tool_end_event(outcome: _ToolOutcome) -> AgentEvent:
    """生成工具执行完成事件。"""

    return AgentEvent(
        type="tool_execution_end",
        tool_call_id=outcome.call.id,
        tool_name=outcome.call.name,
        args=outcome.args_preview,
        result=content_text(outcome.result.content),
        is_error=outcome.result.is_error,
    )


def _tool_update_event(call: ToolCall, result: ToolResult) -> AgentEvent:
    """生成工具执行过程更新事件。"""

    return AgentEvent(
        type="tool_execution_update",
        tool_call_id=call.id,
        tool_name=call.name,
        args=_preview_args(call.arguments),
        result=content_text(result.content),
        is_error=result.is_error,
    )


def _tool_is_available(registry: Registry, name: str) -> bool:
    """判断模型请求的工具是否已注册。"""

    return registry.get(name) is not None


def _visible_tool_definitions(
    tools: list[ToolDefinition],
    visible_tool_names: tuple[str, ...] | None,
) -> list[ToolDefinition]:
    """按运行模式过滤本轮暴露给模型的工具定义。"""

    if visible_tool_names is None:
        return tools
    visible = set(visible_tool_names)
    return [tool for tool in tools if tool.name in visible]


def _build_context(
    conv: Conversation,
    tools: list[ToolDefinition],
    prompt_options: PromptBuildOptions,
    supplemental_instructions: tuple[SupplementalInstruction, ...],
) -> Context:
    """构建单次请求上下文，临时补充消息不写回 Conversation。"""

    resolved_options = _resolve_prompt_options(prompt_options, tools)
    transient_messages = _transient_prompt_messages(supplemental_instructions)
    return Context(
        system_prompt=build_system_prompt(resolved_options),
        messages=[*transient_messages, *conv.messages()],
        tools=tools,
    )


def _resolve_prompt_options(
    options: PromptBuildOptions,
    tools: list[ToolDefinition],
) -> PromptBuildOptions:
    """把本轮工具清单和 cwd 补入提示构建参数。"""

    selected_tools = options.selected_tools or tuple(tool.name for tool in tools)
    tool_snippets = {
        **DEFAULT_TOOL_SNIPPETS,
        **{
            tool.name: tool.prompt_snippet
            for tool in tools
            if tool.prompt_snippet
        },
        **dict(options.tool_snippets),
    }
    prompt_guidelines = _unique_prompt_guidelines(
        [
            guideline
            for tool in tools
            for guideline in tool.prompt_guidelines
        ],
        options.prompt_guidelines,
    )
    return replace(
        options,
        selected_tools=selected_tools,
        tool_snippets=tool_snippets,
        prompt_guidelines=prompt_guidelines,
        cwd=options.cwd or str(Path.cwd()),
    )


def _unique_prompt_guidelines(
    tool_guidelines: list[str],
    option_guidelines: Sequence[str],
) -> tuple[str, ...]:
    """按工具顺序收集 prompt guideline，并保留调用方追加项。"""

    result: list[str] = []
    seen: set[str] = set()
    for guideline in [*tool_guidelines, *option_guidelines]:
        normalized = guideline.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def _transient_prompt_messages(
    supplemental_instructions: tuple[SupplementalInstruction, ...],
) -> list[UserMessage]:
    """把运行时补充指令转换成仅本轮可见的 user 消息。"""

    return [
        UserMessage(content=format_supplemental_instruction(instruction))
        for instruction in supplemental_instructions
    ]


def _append_stop_message(
    conv: Conversation,
    text: str,
    stop_reason: StopReason,
) -> list[AgentEvent]:
    """追加安全停止提示，并生成结束事件。"""

    message = AssistantMessage(content=[text_content(text)], stop_reason="stop")
    conv.add_assistant(text)
    return [
        AgentEvent(type="message_start", message=AssistantMessage()),
        AgentEvent(type="message_update", text=text),
        AgentEvent(type="message_end", message=message),
        AgentEvent(type="turn_end", message=message),
        AgentEvent(type="agent_end", stop_reason=stop_reason),
    ]


def _preview_args(args: dict[str, object]) -> str:
    """生成工具参数的单行预览，避免 UI 中长 JSON 撑开消息。"""

    compact = json.dumps(args or {}, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= TOOL_ARG_PREVIEW_CHARS:
        return compact
    return compact[: TOOL_ARG_PREVIEW_CHARS - 3] + "..."
