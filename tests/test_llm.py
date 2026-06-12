from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentcode.config import ProviderConfig
from agentcode.llm import (
    AssistantMessage,
    Context,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
    assistant_tool_calls,
    create_provider,
)
from agentcode.llm.anthropic_provider import AnthropicProvider
from agentcode.llm.openai_provider import OpenAIProvider
from agentcode.prompt import SYSTEM_PROMPT


class ProviderFactoryTests(unittest.TestCase):
    def test_create_provider_anthropic(self) -> None:
        provider = create_provider(_cfg("anthropic"))
        self.assertIsInstance(provider, AnthropicProvider)

    def test_create_provider_openai(self) -> None:
        provider = create_provider(_cfg("openai"))
        self.assertIsInstance(provider, OpenAIProvider)

    def test_stream_options_keeps_unified_agent_shape(self) -> None:
        options = StreamOptions(
            temperature=0.2,
            max_tokens=123,
            api_key="key",
            transport="auto",
            cache_retention="short",
            session_id="session",
            headers={"x-test": "yes"},
            timeout_ms=1000,
            websocket_connect_timeout_ms=200,
            max_retries=2,
            max_retry_delay_ms=3000,
            metadata={"user_id": "u1"},
        )

        self.assertEqual(options.api_key, "key")
        self.assertEqual(options.transport, "auto")
        self.assertEqual(options.cache_retention, "short")
        self.assertEqual(options.session_id, "session")
        self.assertEqual(options.headers, {"x-test": "yes"})
        self.assertEqual(options.websocket_connect_timeout_ms, 200)
        self.assertEqual(options.max_retry_delay_ms, 3000)
        self.assertEqual(options.metadata, {"user_id": "u1"})


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

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(_deltas(chunks, "thinking_delta"), ["hidden"])
        self.assertEqual(_deltas(chunks, "text_delta"), ["visible"])
        self.assertEqual(chunks[-1].type, "done")
        request = fake_client.messages.requests[0]
        self.assertEqual(request["system"], SYSTEM_PROMPT)
        self.assertEqual(request["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 2048}
        )

    def test_stream_options_map_safe_request_fields(self) -> None:
        fake_client = FakeAnthropicClient([])
        provider = AnthropicProvider(_cfg("anthropic"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    _context("hi"),
                    StreamOptions(temperature=0.2, max_tokens=123),
                )
            )
        )

        self.assertEqual(chunks[-1].type, "done")
        request = fake_client.messages.requests[0]
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["max_tokens"], 123)

    def test_stream_error_includes_diagnostics(self) -> None:
        provider = AnthropicProvider(
            _cfg("anthropic"),
            client=FailingAnthropicClient(RuntimeError("boom")),
        )

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(chunks[-1].type, "error")
        error = chunks[-1].error
        self.assertEqual(error.stop_reason, "error")
        self.assertEqual(error.error_message, "boom")
        self.assertEqual(len(error.diagnostics), 1)
        diagnostic = error.diagnostics[0]
        self.assertEqual(diagnostic.type, "provider_stream_error")
        self.assertIsInstance(diagnostic.timestamp, int)
        self.assertIsNotNone(diagnostic.error)
        self.assertEqual(diagnostic.error.message, "boom")
        self.assertEqual(diagnostic.error.name, "RuntimeError")

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

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(_deltas(chunks, "text_delta"), ["visible"])
        self.assertEqual(chunks[-1].type, "done")

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
                    name="read",
                    input={"path": "spec.md"},
                )
            ],
        )
        fake_client = FakeAnthropicClient([], final_message=final_message)
        provider = AnthropicProvider(_cfg("anthropic"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    _context("read spec", [_tool_definition()]),
                )
            )
        )

        request = fake_client.messages.requests[0]
        self.assertEqual(
            request["tools"],
            [
                {
                    "name": "read",
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
            _tool_calls(chunks),
            [ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})],
        )
        self.assertEqual(chunks[-1].reason, "toolUse")
        self.assertEqual(chunks[-1].type, "done")
        self.assertIsNotNone(chunks[-1].message)
        self.assertEqual(chunks[-1].message.api, "anthropic-messages")
        self.assertEqual(chunks[-1].message.provider, "anthropic")
        self.assertEqual(
            assistant_tool_calls(chunks[-1].message),
            [ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})],
        )

    def test_stream_maps_tool_history_and_disables_thinking(self) -> None:
        call = ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})
        result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="read",
            content=[TextContent(text="content")],
            is_error=False,
        )
        fake_client = FakeAnthropicClient([])
        provider = AnthropicProvider(
            _cfg("anthropic", thinking=True), client=fake_client
        )

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    Context(
                        messages=[
                            AssistantMessage(
                                content=[TextContent(text="Reading."), call]
                            ),
                            result,
                        ]
                    )
                )
            )
        )

        self.assertEqual(chunks[-1].type, "done")
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
                            "name": "read",
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

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(_deltas(chunks, "text_delta"), ["hello"])
        self.assertEqual(chunks[-1].type, "done")
        request = fake_client.chat.completions.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_stream_options_map_safe_request_fields(self) -> None:
        fake_client = FakeOpenAIClient([])
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    _context("hi"),
                    StreamOptions(temperature=0.2, max_tokens=123),
                )
            )
        )

        self.assertEqual(chunks[-1].type, "done")
        request = fake_client.chat.completions.requests[0]
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["max_tokens"], 123)

    def test_stream_error_includes_diagnostics(self) -> None:
        provider = OpenAIProvider(
            _cfg("openai"),
            client=FailingOpenAIClient(RuntimeError("boom")),
        )

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(chunks[-1].type, "error")
        error = chunks[-1].error
        self.assertEqual(error.stop_reason, "error")
        self.assertEqual(error.error_message, "boom")
        self.assertEqual(len(error.diagnostics), 1)
        diagnostic = error.diagnostics[0]
        self.assertEqual(diagnostic.type, "provider_stream_error")
        self.assertIsInstance(diagnostic.timestamp, int)
        self.assertIsNotNone(diagnostic.error)
        self.assertEqual(diagnostic.error.message, "boom")
        self.assertEqual(diagnostic.error.name, "RuntimeError")

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

        chunks = asyncio.run(_collect(provider.stream(_context("hi"))))

        self.assertEqual(_deltas(chunks, "thinking_delta"), ["thinking"])
        self.assertEqual(_deltas(chunks, "text_delta"), ["answer"])

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
                                            name="read",
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
                    _context("read spec", [_tool_definition()]),
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
                        "name": "read",
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
            _tool_calls(chunks),
            [ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})],
        )
        self.assertEqual(chunks[-1].reason, "toolUse")
        self.assertEqual(chunks[-1].type, "done")
        self.assertIsNotNone(chunks[-1].message)
        self.assertEqual(chunks[-1].message.api, "openai-completions")
        self.assertEqual(chunks[-1].message.provider, "openai")

    def test_stream_maps_tool_history(self) -> None:
        call = ToolCall(id="call_1", name="read", arguments={"path": "spec.md"})
        result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="read",
            content=[TextContent(text="content")],
            is_error=False,
        )
        fake_client = FakeOpenAIClient([])
        provider = OpenAIProvider(_cfg("openai"), client=fake_client)

        chunks = asyncio.run(
            _collect(
                provider.stream(
                    Context(
                        messages=[
                            AssistantMessage(
                                content=[TextContent(text="Reading."), call]
                            ),
                            result,
                        ]
                    )
                )
            )
        )

        self.assertEqual(chunks[-1].type, "done")
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
                                "name": "read",
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


class FailingAnthropicMessages:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.requests: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> FakeStream:
        self.requests.append(kwargs)
        raise self._exc


class FakeAnthropicClient:
    def __init__(
        self, events: list[object], final_message: object | None = None
    ) -> None:
        self.messages = FakeAnthropicMessages(events, final_message=final_message)


class FailingAnthropicClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = FailingAnthropicMessages(exc)


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


class FailingOpenAICompletions:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        raise self._exc


class FakeOpenAIChat:
    def __init__(self, events: list[object]) -> None:
        self.completions = FakeOpenAICompletions(events)


class FailingOpenAIChat:
    def __init__(self, exc: Exception) -> None:
        self.completions = FailingOpenAICompletions(exc)


class FakeOpenAIClient:
    def __init__(self, events: list[object]) -> None:
        self.chat = FakeOpenAIChat(events)


class FailingOpenAIClient:
    def __init__(self, exc: Exception) -> None:
        self.chat = FailingOpenAIChat(exc)


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
        name="read",
        description="Read file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


def _context(
    text: str,
    tools: list[ToolDefinition] | None = None,
    system_prompt: str | None = None,
) -> Context:
    return Context(
        messages=[UserMessage(content=text)],
        tools=tools,
        system_prompt=system_prompt,
    )


async def _collect(stream: object) -> list[object]:
    events: list[object] = []
    async for event in stream:  # type: ignore[attr-defined]
        events.append(event)
    return events


def _deltas(events: list[object], event_type: str) -> list[str]:
    return [
        str(getattr(event, "delta", ""))
        for event in events
        if getattr(event, "type", None) == event_type
    ]


def _tool_calls(events: list[object]) -> list[ToolCall]:
    return [
        event.tool_call
        for event in events
        if getattr(event, "type", None) == "toolcall_end"
        and event.tool_call is not None
    ]


if __name__ == "__main__":
    unittest.main()
