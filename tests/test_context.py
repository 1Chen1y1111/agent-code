"""
上下文治理模块的单元测试。

覆盖工具结果外部化、artifact 路径策略、会话压缩和 provider 溢出识别。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentcode.context import ContextManager, ContextSettings, is_context_overflow
from agentcode.conversation import Conversation
from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    StartEvent,
    StreamOptions,
    TextContent,
    ToolCall,
    Usage,
    message_text,
)
from agentcode.permission import path_stays_in_project
from agentcode.tool import content_text, text_result


def test_default_artifact_dir_stays_inside_project_root(tmp_path: Path) -> None:
    """默认 artifact 目录位于项目内，方便 read 工具后续重读。"""

    manager = ContextManager(project_root=tmp_path, session_id="session")

    expected = tmp_path / ".agentcode" / "context-artifacts" / "session"
    assert manager.artifact_dir == expected
    assert path_stays_in_project(tmp_path, expected / "artifact.txt")


def test_externalize_tool_result_writes_artifact_and_keeps_stable_preview(
    tmp_path: Path,
) -> None:
    """超大工具结果会落盘，active 历史只保留稳定预览和重读路径。"""

    manager = ContextManager(
        ContextSettings(
            max_inline_tool_result_chars=10,
            tool_result_preview_chars=8,
            artifact_root=tmp_path,
        ),
        session_id="session",
    )
    call = ToolCall(id="call-1", name="read", arguments={"path": "big.py"})
    active, archive = manager.externalize_tool_result(call, text_result("0123456789abc"))

    assert content_text(archive.content) == "0123456789abc"
    active_text = content_text(active.content)
    assert "[agentcode tool result externalized]" in active_text
    assert "tool: read" in active_text
    assert "preview_chars:" in active_text
    artifact_path = Path(active.details["context_artifact"]["path"])
    assert artifact_path.read_text(encoding="utf-8") == "0123456789abc"

    active_again, _ = manager.externalize_tool_result(call, text_result("0123456789abc"))
    assert content_text(active_again.content) == active_text


@pytest.mark.asyncio
async def test_compact_conversation_replaces_active_history_but_keeps_archive(
    tmp_path: Path,
) -> None:
    """压缩只替换模型可见 active 历史，原始 archive 仍保留用户消息。"""

    conversation = Conversation()
    conversation.add_user("请修改 a.py")
    conversation.add_assistant("我会先读取")
    conversation.add_tool_results([])
    conversation.add_user("继续")
    provider = FakeProvider([_assistant_events("摘要内容")])
    manager = ContextManager(
        ContextSettings(keep_recent_tokens=1, artifact_root=tmp_path),
        session_id="session",
    )

    report = await manager.compact_conversation(conversation, provider, [])

    assert report is not None
    assert report.summarized_messages > 0
    active = conversation.messages()
    assert active[0].role == "user"
    assert "摘要内容" in message_text(active[0])
    assert any(message_text(message) == "请修改 a.py" for message in conversation.archive_messages())


def test_is_context_overflow_uses_error_patterns_and_usage() -> None:
    """上下文溢出检测覆盖 provider 错误文本和静默 usage 超窗。"""

    error = AssistantMessage(
        stop_reason="error",
        error_message="context_length_exceeded: prompt too large",
    )
    ok = AssistantMessage(
        stop_reason="error",
        error_message="rate limit: too many tokens per minute",
    )
    silent = AssistantMessage(
        stop_reason="stop",
        usage=Usage(input=101, output=1, total_tokens=102),
    )

    assert is_context_overflow(error, 100)
    assert not is_context_overflow(ok, 100)
    assert is_context_overflow(silent, 100)


class FakeProvider:
    """测试用 provider，按脚本返回事件流并记录请求。"""

    def __init__(self, scripts: list[list[AssistantMessageEvent]]) -> None:
        """保存事件脚本和 provider 元数据。"""

        self.api = "openai-completions"
        self.name = "fake"
        self.model = "fake-model"
        self.context_window = 128000
        self._scripts = scripts
        self.requests: list[Context] = []
        self.stream_options: list[StreamOptions | None] = []

    async def stream(
        self,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """记录请求并产出下一段脚本。"""

        self.requests.append(context)
        self.stream_options.append(options)
        for event in self._scripts.pop(0):
            yield event


def _assistant_events(text: str) -> list[AssistantMessageEvent]:
    """创建只有文本的 assistant 完成事件。"""

    content: list[AssistantContent] = [TextContent(text=text)]
    message = AssistantMessage(
        content=content,
        api="fake",
        provider="fake",
        model="fake-model",
        stop_reason="stop",
    )
    return [StartEvent(partial=AssistantMessage()), DoneEvent(reason="stop", message=message)]
