"""
长期记忆笔记模块的单元测试。

覆盖笔记去重保存、关键词召回和 LLM 提取结果解析。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentcode.llm import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    StartEvent,
    StreamOptions,
    TextContent,
    UserMessage,
)
from agentcode.memory import MemoryExtractor, MemoryNote, MemoryStore


def test_memory_store_saves_dedupes_and_recalls_notes(tmp_path: Path) -> None:
    """笔记按分类落盘，重复内容不会再次保存。"""

    store = MemoryStore(
        ".agentcode/memory",
        project_root=tmp_path,
        user_notes_dir=tmp_path / "user-memory",
    )
    note = MemoryNote(
        category="project",
        content="项目要求中文注释",
        confidence=0.9,
        source_session_id="20260615-120000-abcd",
        created_at="2026-06-15T12:00:00",
    )

    assert store.save_notes([note, note]) == 1
    recalled = store.relevant_notes("请保持中文注释")

    assert len(recalled) == 1
    assert recalled[0].category == "project"
    assert recalled[0].content == "项目要求中文注释"


@pytest.mark.asyncio
async def test_memory_extractor_parses_provider_json() -> None:
    """自动笔记提取器会解析 provider 返回的 JSON 数组。"""

    provider = FakeProvider(
        "[{\"category\":\"user\",\"content\":\"用户喜欢简洁回答\",\"confidence\":0.8}]"
    )
    extractor = MemoryExtractor(max_tokens=100)

    notes = await extractor.extract(
        provider,
        [UserMessage(content="以后请简洁回答")],
        session_id="20260615-120000-abcd",
    )

    assert len(notes) == 1
    assert notes[0].category == "user"
    assert notes[0].content == "用户喜欢简洁回答"
    assert provider.requests[0].messages[0].role == "user"
    assert provider.options[0].max_tokens == 100


class FakeProvider:
    """测试用 provider，返回固定文本 completion。"""

    def __init__(self, text: str) -> None:
        """保存要返回的文本和请求记录。"""

        self.api = "fake"
        self.name = "Fake"
        self.model = "fake-model"
        self.context_window = 100_000
        self.text = text
        self.requests: list[Context] = []
        self.options: list[StreamOptions] = []

    async def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """记录请求并产出一次 done 事件。"""

        self.requests.append(context)
        self.options.append(options or StreamOptions())
        message = AssistantMessage(content=[TextContent(text=self.text)])
        yield StartEvent(partial=AssistantMessage())
        yield DoneEvent(reason="stop", message=message)
