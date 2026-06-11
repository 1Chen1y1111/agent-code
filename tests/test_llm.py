from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentcode.config import ProviderConfig
from agentcode.llm import Message, ToolCall, ToolDefinition, ToolResult, new_provider
from agentcode.llm.anthropic_provider import AnthropicProvider
from agentcode.llm.openai_provider import OpenAIProvider
from agentcode.prompt import SYSTEM_PROMPT


class ProviderFactoryTests(unittest.TestCase):
    def test_new_provider_anthropic(self) -> None:
        provider = new_provider(_cfg("anthropic"))
        self.assertIsInstance(provider, AnthropicProvider)

    def test_new_provider_openai(self) -> None:
        provider = new_provider(_cfg("openai"))
        self.assertIsInstance(provider, OpenAIProvider)


class AnthropicProviderTests(unittest.TestCase):
    def test_stream_sends_system_history_and_thinking(self) -> None:
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="hidden"),
            ),
            SimpleNamespace(type="text", text="visible", snapshot="visible"),
        ]
        fake_client = FakeAnthropicClient(events)
        provider = AnthropicProvider(
            _cfg("anthropic", thinking=True), client=fake_client
        )

        chunks = asyncio.run(_collect(provider.stream([Message("user", "hi")])))

        self.assertEqual(
            [event.thinking for event in chunks if event.thinking], ["hidden"]
        )
        self.assertEqual([event.text for event in chunks if event.text], ["visible"])
        self.assertTrue(chunks[-1].done)
        request = fake_client.messages.requests[0]
        self.assertEqual(request["system"], SYSTEM_PROMPT)
        self.assertEqual(request["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 2048}
        )

    def test_stream_ignores_raw_delta_to_avoid_duplicate_text(self) -> None:
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="visible"),
            ),
            SimpleNamespace(type="text", text="visible", snapshot="visible"),
        ]
        fake_client = FakeAnthropicClient(events)
        provider = AnthropicProvider(_cfg("anthropic"), client=fake_client)

        chunks = asyncio.run(_collect(provider.stream([Message("user", "hi")])))

        self.assertEqual([event.text for event in chunks if event.text], ["visible"])
        self.assertTrue(chunks[-1].done)

    def test_constructor_passes_base_url_when_present(self) -> None:
        with patch("agentcode.llm.anthropic_provider.AsyncAnthropic", FakeAnthropicSDK):
            AnthropicProvider(
                _cfg("anthropic", base_url="https://anthropic.example.com")
            )

        self.assertEqual(
            FakeAnthropicSDK.last_kwargs["base_url"], "https://anthropic.example.com"
        )
        self.assertEqual(FakeAnthropicSDK.last_kwargs["api_key"], "test-key")

    def test_stream_sends_tools_and_extracts_tool_calls(self) -> None:
        final_message = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="read_file",
                    input={"path": "spec.md"},
                )
            ],
        )
        fake_client = FakeAnthropicClient([], final_message=final_message)
        provider = AnthropicProvider(_cfg("anthropic"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    [Message("user", "read spec")],
                    [_tool_definition()],
                )
            )
        )

        request = fake_client.messages.requests[0]
        self.assertEqual(
            request["tools"],
            [
                {
                    "name": "read_file",
                    "description": "Read file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )
        self.assertEqual(
            [event.tool_calls for event in chunks if event.tool_calls],
            [[ToolCall(id="call_1", name="read_file", input='{"path": "spec.md"}')]],
        )
        self.assertTrue(chunks[-1].done)

    def test_stream_maps_tool_history_and_disables_thinking(self) -> None:
        call = ToolCall(id="call_1", name="read_file", input='{"path":"spec.md"}')
        result = ToolResult(tool_call_id="call_1", content="content", is_error=False)
        fake_client = FakeAnthropicClient([])
        provider = AnthropicProvider(
            _cfg("anthropic", thinking=True), client=fake_client
        )

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    [
                        Message("assistant", "Reading.", tool_calls=[call]),
                        Message("tool", tool_results=[result]),
                    ]
                )
            )
        )

        self.assertTrue(chunks[-1].done)
        request = fake_client.messages.requests[0]
        self.assertNotIn("thinking", request)
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Reading."},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read_file",
                            "input": {"path": "spec.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "content",
                            "is_error": False,
                        }
                    ],
                },
            ],
        )


class OpenAIProviderTests(unittest.TestCase):
    def test_stream_sends_system_history(self) -> None:
        fake_client = FakeOpenAIClient(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
                )
            ]
        )
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(_collect(provider.stream([Message("user", "hi")])))

        self.assertEqual([event.text for event in chunks if event.text], ["hello"])
        self.assertTrue(chunks[-1].done)
        request = fake_client.chat.completions.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_stream_extracts_reasoning_delta_separately(self) -> None:
        fake_client = FakeOpenAIClient(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                reasoning_content="thinking",
                                content="answer",
                            )
                        )
                    ]
                )
            ]
        )
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(_collect(provider.stream([Message("user", "hi")])))

        self.assertEqual(
            [event.thinking for event in chunks if event.thinking], ["thinking"]
        )
        self.assertEqual([event.text for event in chunks if event.text], ["answer"])

    def test_constructor_passes_base_url_when_present(self) -> None:
        with patch("agentcode.llm.openai_provider.AsyncOpenAI", FakeOpenAISDK):
            OpenAIProvider(_cfg("openai", base_url="https://openai.example.com"))

        self.assertEqual(
            FakeOpenAISDK.last_kwargs["base_url"], "https://openai.example.com"
        )
        self.assertEqual(FakeOpenAISDK.last_kwargs["api_key"], "test-key")

    def test_stream_sends_tools_and_extracts_tool_calls(self) -> None:
        fake_client = FakeOpenAIClient(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1",
                                        function=SimpleNamespace(
                                            name="read_file",
                                            arguments='{"path"',
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id=None,
                                        function=SimpleNamespace(
                                            name=None,
                                            arguments=':"spec.md"}',
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                ),
            ]
        )
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    [Message("user", "read spec")],
                    [_tool_definition()],
                )
            )
        )

        request = fake_client.chat.completions.requests[0]
        self.assertEqual(
            request["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )
        self.assertEqual(
            [event.tool_calls for event in chunks if event.tool_calls],
            [[ToolCall(id="call_1", name="read_file", input='{"path":"spec.md"}')]],
        )
        self.assertTrue(chunks[-1].done)

    def test_stream_maps_tool_history(self) -> None:
        call = ToolCall(id="call_1", name="read_file", input='{"path":"spec.md"}')
        result = ToolResult(tool_call_id="call_1", content="content", is_error=False)
        fake_client = FakeOpenAIClient([])
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    [
                        Message("assistant", "Reading.", tool_calls=[call]),
                        Message("tool", tool_results=[result]),
                    ]
                )
            )
        )

        self.assertTrue(chunks[-1].done)
        request = fake_client.chat.completions.requests[0]
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "assistant",
                    "content": "Reading.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"spec.md"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "content",
                },
            ],
        )


class FakeStream:
    def __init__(
        self, events: list[object], final_message: object | None = None
    ) -> None:
        self._events = events
        self._final_message = final_message or SimpleNamespace(stop_reason="end_turn")

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def __aiter__(self) -> object:
        return self._iterate()

    async def _iterate(self) -> object:
        for event in self._events:
            yield event

    async def get_final_message(self) -> object:
        return self._final_message


class FakeAnthropicMessages:
    def __init__(
        self, events: list[object], final_message: object | None = None
    ) -> None:
        self._events = events
        self._final_message = final_message
        self.requests: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> FakeStream:
        self.requests.append(kwargs)
        return FakeStream(self._events, final_message=self._final_message)


class FakeAnthropicClient:
    def __init__(
        self, events: list[object], final_message: object | None = None
    ) -> None:
        self.messages = FakeAnthropicMessages(events, final_message=final_message)


class FakeAnthropicSDK(FakeAnthropicClient):
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        super().__init__([])


class FakeOpenAICompletions:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return FakeAsyncIterator(self._events)


class FakeOpenAIChat:
    def __init__(self, events: list[object]) -> None:
        self.completions = FakeOpenAICompletions(events)


class FakeOpenAIClient:
    def __init__(self, events: list[object]) -> None:
        self.chat = FakeOpenAIChat(events)


class FakeOpenAISDK(FakeOpenAIClient):
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        super().__init__([])


class FakeAsyncIterator:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def __aiter__(self) -> object:
        return self._iterate()

    async def _iterate(self) -> object:
        for event in self._events:
            yield event


def _cfg(
    protocol: str, thinking: bool = False, base_url: str | None = None
) -> ProviderConfig:
    return ProviderConfig(
        name=f"{protocol} provider",
        protocol=protocol,  # type: ignore[arg-type]
        api_key="test-key",
        model="test-model",
        base_url=base_url,
        thinking=thinking,
    )


def _tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


async def _collect(stream: object) -> list[object]:
    events: list[object] = []
    async for event in stream:  # type: ignore[attr-defined]
        events.append(event)
    return events


if __name__ == "__main__":
    unittest.main()
