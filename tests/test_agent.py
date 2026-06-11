from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentcode.agent import Agent, Phase, SINGLE_TOOL_ROUND_LIMIT_MESSAGE
from agentcode.conversation import Conversation
from agentcode.llm import Message, StreamEvent, ToolCall, ToolDefinition
from agentcode.tool import Registry, Result


@pytest.mark.asyncio
async def test_agent_runs_single_tool_round_and_final_answer() -> None:
    conv = Conversation()
    conv.add_user("read it")
    provider = FakeProvider(
        [
            [
                StreamEvent(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="read_file",
                            input='{"path":"note.txt"}',
                        )
                    ]
                ),
                StreamEvent(done=True),
            ],
            [StreamEvent(text="文件已读取"), StreamEvent(done=True)],
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("read_file", "content"))

    events = [event async for event in Agent(provider, registry).run(conv)]

    tool_events = [event.tool for event in events if event.tool is not None]
    assert [tool.phase for tool in tool_events if tool is not None] == [
        Phase.START,
        Phase.END,
    ]
    assert [event.text for event in events if event.text] == ["文件已读取"]
    assert events[-1].done
    assert [message.role for message in conv.messages()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert conv.messages()[-1] == Message(role="assistant", content="文件已读取")
    assert registry.calls == [("read_file", '{"path":"note.txt"}')]
    assert len(provider.requests) == 2
    assert provider.requests[0][1][0].name == "read_file"


@pytest.mark.asyncio
async def test_agent_does_not_execute_second_round_tool_calls() -> None:
    conv = Conversation()
    conv.add_user("read twice")
    provider = FakeProvider(
        [
            [
                StreamEvent(
                    tool_calls=[ToolCall(id="call_1", name="read_file", input="{}")]
                ),
                StreamEvent(done=True),
            ],
            [
                StreamEvent(
                    tool_calls=[ToolCall(id="call_2", name="read_file", input="{}")]
                ),
                StreamEvent(done=True),
            ],
        ]
    )
    registry = RecordingRegistry()
    registry.register(FakeTool("read_file", "content"))

    events = [event async for event in Agent(provider, registry).run(conv)]

    assert registry.calls == [("read_file", "{}")]
    assert [event.text for event in events if event.text] == [
        SINGLE_TOOL_ROUND_LIMIT_MESSAGE
    ]
    assert conv.messages()[-1].content == SINGLE_TOOL_ROUND_LIMIT_MESSAGE


@pytest.mark.asyncio
async def test_agent_streams_thinking_without_storing_it_in_history() -> None:
    conv = Conversation()
    conv.add_user("explain")
    provider = FakeProvider(
        [
            [
                StreamEvent(thinking="先分析"),
                StreamEvent(text="答案"),
                StreamEvent(done=True),
            ]
        ]
    )
    registry = RecordingRegistry()

    events = [event async for event in Agent(provider, registry).run(conv)]

    assert [event.thinking for event in events if event.thinking] == ["先分析"]
    assert [event.text for event in events if event.text] == ["答案"]
    assert conv.messages()[-1] == Message(role="assistant", content="答案")


class FakeProvider:
    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.name = "Fake"
        self.model = "fake-model"
        self._scripts = scripts
        self.requests: list[tuple[list[Message], list[ToolDefinition]]] = []

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition] | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append((msgs, tools or []))
        script = self._scripts.pop(0)
        for event in script:
            yield event


class FakeTool:
    def __init__(self, name: str, content: str) -> None:
        self._name = name
        self._content = content

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "fake"

    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    async def execute(self, args: str) -> Result:
        return Result(self._content)


class RecordingRegistry(Registry):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    async def execute(self, name: str, args: str, timeout: float = 30.0) -> Result:
        self.calls.append((name, args))
        return await super().execute(name, args, timeout=timeout)
