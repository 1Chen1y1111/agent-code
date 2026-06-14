from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentcode.agent import (
    MAX_ITERATIONS_MESSAGE,
    UNKNOWN_TOOL_LIMIT_MESSAGE,
    Agent,
    AgentRunOptions,
)
from agentcode.conversation import Conversation
from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    DoneStopReason,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    Usage,
    message_text,
)
from agentcode.prompt import (
    PromptBuildOptions,
    PromptContextFile,
    SupplementalInstruction,
)
from agentcode.permission import PermissionPolicy, ToolCategory, load_permission_config
from agentcode.session import AgentSession
from agentcode.tool import (
    BaseTool,
    ExecutionMode,
    Registry,
    ToolResult,
    ToolUpdate,
    text_result,
)


@pytest.mark.asyncio
async def test_agent_loops_until_model_stops_requesting_tools() -> None:
    conv = Conversation()
    conv.add_user("read and grep")
    provider = FakeProvider(
        [
            [
                *_assistant_events(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="read",
                            arguments={"path": "note.txt"},
                        )
                    ]
                ),
            ],
            [
                *_assistant_events(
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="grep",
                            arguments={"pattern": "todo"},
                        )
                    ]
                ),
            ],
            _assistant_events(text="最终答案"),
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("read", "file content", "parallel"))
    registry.register(FakeTool("grep", "grep content", "parallel"))

    events = [event async for event in Agent(provider, registry).run(conv)]

    assert [event.text for event in events if event.text] == ["最终答案"]
    assert registry.calls == [
        ("read", {"path": "note.txt"}),
        ("grep", {"pattern": "todo"}),
    ]
    assert len(provider.requests) == 3
    assert [message.role for message in conv.messages()] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
        "toolResult",
        "assistant",
    ]
    tool_messages = [
        message for message in conv.messages() if message.role == "toolResult"
    ]
    assert [message.tool_name for message in tool_messages] == ["read", "grep"]
    assistant_messages = [
        message for message in conv.messages() if message.role == "assistant"
    ]
    assert [message.stop_reason for message in assistant_messages] == [
        "toolUse",
        "toolUse",
        "stop",
    ]
    assert message_text(conv.messages()[-1]) == "最终答案"
    assert events[-1].type == "agent_end"
    assert events[-1].stop_reason == "completed"


@pytest.mark.asyncio
async def test_agent_stops_at_iteration_limit_after_tool_result() -> None:
    conv = Conversation()
    conv.add_user("loop")
    provider = FakeProvider(
        [
            [
                *_assistant_events(tool_calls=[ToolCall(id="call_1", name="read")]),
            ]
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("read", "content", "parallel"))

    events = [
        event
        async for event in Agent(provider, registry).run(
            conv, AgentRunOptions(max_iterations=1)
        )
    ]

    assert registry.calls == [("read", {})]
    assert len(provider.requests) == 1
    assert [event.text for event in events if event.text] == [MAX_ITERATIONS_MESSAGE]
    assert message_text(conv.messages()[-1]) == MAX_ITERATIONS_MESSAGE
    assert events[-1].stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_agent_stops_after_repeated_unknown_tools() -> None:
    conv = Conversation()
    conv.add_user("call missing")
    provider = FakeProvider(
        [
            [
                *_assistant_events(tool_calls=[ToolCall(id="call_1", name="missing")]),
            ],
            [
                *_assistant_events(tool_calls=[ToolCall(id="call_2", name="missing")]),
            ],
        ]
    )
    registry = RecordingRegistry()

    events = [
        event
        async for event in Agent(provider, registry).run(
            conv, AgentRunOptions(max_unknown_tools=2)
        )
    ]

    tool_errors = [
        event.result for event in events if event.type == "tool_execution_end"
    ]
    assert tool_errors == [
        "未知工具: missing",
        "未知工具: missing",
    ]
    assert registry.calls == []
    assert len(provider.requests) == 2
    assert [event.text for event in events if event.text] == [
        UNKNOWN_TOOL_LIMIT_MESSAGE
    ]
    assert events[-1].stop_reason == "unknown_tool_limit"


@pytest.mark.asyncio
async def test_agent_runs_parallel_tools_concurrently() -> None:
    conv = Conversation()
    conv.add_user("parallel")
    provider = FakeProvider(
        [
            [
                *_assistant_events(
                    tool_calls=[
                        ToolCall(id="call_1", name="read"),
                        ToolCall(id="call_2", name="grep"),
                    ]
                ),
            ],
            _assistant_events(text="done"),
        ]
    )
    probe = ConcurrencyProbe()
    registry = RecordingRegistry()
    registry.register(ProbeTool("read", "parallel", probe))
    registry.register(ProbeTool("grep", "parallel", probe))

    await _collect(Agent(provider, registry).run(conv))

    assert probe.max_active == 2
    assert registry.calls == [("read", {}), ("grep", {})]


@pytest.mark.asyncio
async def test_agent_runs_side_effect_tools_sequentially() -> None:
    conv = Conversation()
    conv.add_user("sequential")
    provider = FakeProvider(
        [
            [
                *_assistant_events(
                    tool_calls=[
                        ToolCall(id="call_1", name="write", arguments={"path": "a"}),
                        ToolCall(
                            id="call_2",
                            name="bash",
                            arguments={"command": "x"},
                        ),
                    ]
                ),
            ],
            _assistant_events(text="done"),
        ]
    )
    probe = ConcurrencyProbe()
    registry = RecordingRegistry()
    registry.register(ProbeTool("write", "sequential", probe))
    registry.register(ProbeTool("bash", "sequential", probe))

    await _collect(Agent(provider, registry).run(conv))

    assert probe.max_active == 1
    assert probe.order == ["write", "bash"]
    assert registry.calls == [
        ("write", {"path": "a"}),
        ("bash", {"command": "x"}),
    ]


@pytest.mark.asyncio
async def test_agent_forwards_tool_updates_and_terminates_when_requested() -> None:
    conv = Conversation()
    conv.add_user("update")
    provider = FakeProvider(
        [
            [
                *_assistant_events(tool_calls=[ToolCall(id="call_1", name="stop")]),
            ]
        ]
    )
    registry = RecordingRegistry()
    registry.register(UpdatingTerminatingTool())

    events = [event async for event in Agent(provider, registry).run(conv)]

    assert [
        event.result for event in events if event.type == "tool_execution_update"
    ] == ["partial"]
    assert events[-1].type == "agent_end"
    assert events[-1].stop_reason == "tool_terminated"
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_agent_streams_usage_and_thinking_without_storing_thinking() -> None:
    conv = Conversation()
    conv.add_user("explain")
    usage = Usage(input=3, output=5, total_tokens=8)
    provider = FakeProvider(
        [_assistant_events(thinking="先分析", text="答案", usage=usage)]
    )
    registry = RecordingRegistry()

    events = [event async for event in Agent(provider, registry).run(conv)]

    assert [event.thinking for event in events if event.thinking] == ["先分析"]
    assert [event.text for event in events if event.text] == ["答案"]
    assert [event.usage for event in events if event.usage is not None] == [usage]
    assert message_text(conv.messages()[-1]) == "答案"
    assert conv.messages()[-1].usage == usage
    assert conv.messages()[-1].stop_reason == "stop"


@pytest.mark.asyncio
async def test_agent_session_emits_user_events_and_keeps_history() -> None:
    provider = FakeProvider([_assistant_events(text="你好")])
    session = AgentSession(provider, RecordingRegistry())

    events = [event async for event in session.prompt("hi")]

    assert [event.type for event in events[:4]] == [
        "agent_start",
        "message_start",
        "message_end",
        "turn_start",
    ]
    assert events[2].message is not None
    assert events[2].message.role == "user"
    assert message_text(events[2].message) == "hi"
    assert [
        (message.role, message_text(message)) for message in session.messages()
    ] == [
        ("user", "hi"),
        ("assistant", "你好"),
    ]


@pytest.mark.asyncio
async def test_agent_session_passes_prompt_context_files_to_system_prompt() -> None:
    provider = FakeProvider([_assistant_events(text="你好")])
    session = AgentSession(
        provider,
        RecordingRegistry(),
        PromptBuildOptions(
            context_files=(
                PromptContextFile(path="/tmp/AGENTS.md", content="始终中文回答"),
            )
        ),
    )

    await _collect(session.prompt("hi"))

    system_prompt = provider.requests[0].system_prompt or ""
    assert '<project_instructions path="/tmp/AGENTS.md">' in system_prompt
    assert "始终中文回答" in system_prompt
    assert "Current date:" in system_prompt
    assert "Current working directory:" in system_prompt


@pytest.mark.asyncio
async def test_agent_injects_supplemental_context_without_persisting_it() -> None:
    conv = Conversation()
    conv.add_user("hi")
    provider = FakeProvider([_assistant_events(text="ok")])
    registry = RecordingRegistry()

    events = [
        event
        async for event in Agent(provider, registry).run(
            conv,
            AgentRunOptions(
                supplemental_instructions=(
                    SupplementalInstruction(source="test", content="Remember X"),
                ),
                session_id="session-1",
            ),
        )
    ]

    assert events[-1].stop_reason == "completed"
    request = provider.requests[0]
    assert request.system_prompt is not None
    assert "Available tools:" in request.system_prompt
    assert "Current date:" in request.system_prompt
    assert "Current working directory:" in request.system_prompt
    assert len(request.messages) == 2
    assert request.messages[0].role == "user"
    assert 'source="test"' in str(request.messages[0].content)
    assert "Remember X" in str(request.messages[0].content)
    assert message_text(request.messages[1]) == "hi"
    assert [(message.role, message_text(message)) for message in conv.messages()] == [
        ("user", "hi"),
        ("assistant", "ok"),
    ]
    assert provider.stream_options[0] is not None
    assert provider.stream_options[0].cache_retention == "short"
    assert provider.stream_options[0].session_id == "session-1"


@pytest.mark.asyncio
async def test_agent_system_prompt_uses_tool_prompt_metadata() -> None:
    conv = Conversation()
    conv.add_user("hi")
    provider = FakeProvider([_assistant_events(text="ok")])
    registry = RecordingRegistry()
    registry.register(PromptMetadataTool())

    await _collect(Agent(provider, registry).run(conv))

    system_prompt = provider.requests[0].system_prompt or ""
    assert "- meta: Short prompt snippet" in system_prompt
    assert "Use meta carefully." in system_prompt
    assert "Provider-only description" not in system_prompt


@pytest.mark.asyncio
async def test_agent_denies_tool_without_executing_and_continues(tmp_path) -> None:
    local = tmp_path / "permissions.local.yaml"
    local.write_text(
        """
deny:
  - Bash(echo blocked)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=local,
        ),
    )
    conv = Conversation()
    conv.add_user("run")
    provider = FakeProvider(
        [
            _assistant_events(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="bash",
                        arguments={"command": "echo blocked"},
                    )
                ]
            ),
            _assistant_events(text="换个办法"),
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("bash", "should not run", "sequential"))

    events = [
        event
        async for event in Agent(provider, registry).run(
            conv,
            AgentRunOptions(permission_policy=policy),
        )
    ]

    assert registry.calls == []
    tool_results = [
        message for message in conv.messages() if message.role == "toolResult"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "call_1"
    assert tool_results[0].is_error
    assert tool_results[0].details["permission_denied"] is True
    assert tool_results[0].details["source"] == "rule"
    assert "权限拒绝" in message_text(tool_results[0])
    assert events[-1].stop_reason == "completed"
    assert message_text(conv.messages()[-1]) == "换个办法"


@pytest.mark.asyncio
async def test_agent_asks_permission_and_can_remember_allow(tmp_path) -> None:
    local = tmp_path / "permissions.local.yaml"
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=local,
        ),
    )
    conv = Conversation()
    conv.add_user("write")
    provider = FakeProvider(
        [
            _assistant_events(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="write",
                        arguments={"path": "note.txt"},
                    )
                ]
            ),
            _assistant_events(text="done"),
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("write", "wrote", "sequential"))
    requests = []

    async def approve(request) -> str:
        requests.append(request)
        return "allow_always"

    await _collect(
        Agent(provider, registry).run(
            conv,
            AgentRunOptions(
                permission_policy=policy,
                permission_approver=approve,  # type: ignore[arg-type]
            ),
        )
    )

    assert registry.calls == [("write", {"path": "note.txt"})]
    assert requests[0].exact_rule == "Write(note.txt)"
    assert "Write(note.txt)" in local.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_plan_mode_only_exposes_readonly_tools() -> None:
    """plan 模式默认只暴露内置只读工具。"""

    provider = FakeProvider([_assistant_events(text="计划")])
    registry = RecordingRegistry()
    registry.register(FakeTool("read", "read", "parallel"))
    registry.register(FakeTool("write", "write", "sequential"))
    registry.register(FakeTool("bash", "bash", "sequential"))
    session = AgentSession(
        provider,
        registry,
        permission_mode=lambda: "plan",
    )

    await _collect(session.prompt("plan"))

    request = provider.requests[0]
    assert request.tools is not None
    assert [tool.name for tool in request.tools] == ["read"]
    assert "plan mode" in str(request.messages[0].content)


@pytest.mark.asyncio
async def test_plan_mode_exposes_dynamic_readonly_tools() -> None:
    """plan 模式允许暴露由 Registry 声明为 readonly 的动态工具。"""

    provider = FakeProvider([_assistant_events(text="计划")])
    registry = RecordingRegistry()
    registry.register(FakeTool("read", "read", "parallel"))
    registry.register(
        FakeTool("mcp__github__get_issue", "issue", "parallel", "readonly")
    )
    registry.register(
        FakeTool("mcp__github__create_issue", "issue", "sequential", "command")
    )
    session = AgentSession(
        provider,
        registry,
        permission_mode=lambda: "plan",
    )

    await _collect(session.prompt("plan"))

    request = provider.requests[0]
    assert request.tools is not None
    assert [tool.name for tool in request.tools] == [
        "read",
        "mcp__github__get_issue",
    ]


async def _collect(stream: AsyncIterator[object]) -> list[object]:
    events: list[object] = []
    async for event in stream:
        events.append(event)
    return events


class FakeProvider:
    def __init__(self, scripts: list[list[AssistantMessageEvent]]) -> None:
        self.api = "fake"
        self.name = "Fake"
        self.model = "fake-model"
        self._scripts = scripts
        self.requests: list[Context] = []
        self.stream_options: list[StreamOptions | None] = []

    async def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.requests.append(context)
        self.stream_options.append(options)
        script = self._scripts.pop(0)
        for event in script:
            yield event


class FakeTool(BaseTool):
    def __init__(
        self,
        name: str,
        content: str,
        mode: ExecutionMode,
        category: ToolCategory | None = None,
    ) -> None:
        self._name = name
        self._content = content
        self._mode = mode
        self._category = category

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "fake"

    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    def execution_mode(self) -> ExecutionMode:
        return self._mode

    def permission_category(self) -> ToolCategory | None:
        return self._category

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        return text_result(self._content)


class ConcurrencyProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []

    async def run(self, name: str) -> ToolResult:
        self.order.append(name)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return text_result(f"{name} done")


class ProbeTool(FakeTool):
    def __init__(self, name: str, mode: ExecutionMode, probe: ConcurrencyProbe) -> None:
        super().__init__(name, f"{name} done", mode)
        self._probe = probe

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        return await self._probe.run(self.name())


class PromptMetadataTool(FakeTool):
    def __init__(self) -> None:
        super().__init__("meta", "ok", "parallel")

    def description(self) -> str:
        """返回只应进入 provider tools 的说明。"""

        return "Provider-only description"

    def prompt_snippet(self) -> str:
        """返回只应进入 system prompt 工具列表的摘要。"""

        return "Short prompt snippet"

    def prompt_guidelines(self) -> list[str]:
        """返回只应进入 system prompt guidelines 的行为约束。"""

        return ["Use meta carefully."]


class UpdatingTerminatingTool(FakeTool):
    def __init__(self) -> None:
        super().__init__("stop", "done", "sequential")

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        if on_update is not None:
            on_update(text_result("partial"))
        return text_result("done", terminate=True)


class RecordingRegistry(Registry):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self,
        tool_call_id: str,
        name: str,
        args: dict[str, Any],
        timeout: float = 30.0,
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        self.calls.append((name, args))
        return await super().execute(
            tool_call_id,
            name,
            args,
            timeout=timeout,
            on_update=on_update,
        )


def _assistant_events(
    text: str = "",
    thinking: str = "",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> list[AssistantMessageEvent]:
    """创建测试用统一 assistant 事件流。"""

    content: list[AssistantContent] = []
    events: list[AssistantMessageEvent] = [StartEvent(partial=AssistantMessage())]
    if thinking:
        content.append(ThinkingContent(thinking=thinking))
        events.append(
            ThinkingDeltaEvent(
                content_index=len(content) - 1,
                delta=thinking,
                partial=AssistantMessage(content=list(content)),
            )
        )
    if text:
        content.append(TextContent(text=text))
        events.append(
            TextDeltaEvent(
                content_index=len(content) - 1,
                delta=text,
                partial=AssistantMessage(content=list(content)),
            )
        )
    content.extend(tool_calls or [])
    reason: DoneStopReason = "toolUse" if tool_calls else "stop"
    message = AssistantMessage(
        content=content,
        api="fake",
        provider="Fake",
        model="fake-model",
        usage=usage or Usage(),
        stop_reason=reason,
    )
    events.append(DoneEvent(reason=reason, message=message))
    return events
