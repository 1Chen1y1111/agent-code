"""
会话存档模块的单元测试。

覆盖 session id、JSONL 追加恢复、坏行容错、孤立工具结果修剪和过期清理。
"""

from __future__ import annotations

from datetime import datetime
import os
import time
from pathlib import Path

from agentcode.llm import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from agentcode.session_store import SessionStore, generate_session_id
from agentcode.tool import text_result


def test_generate_session_id_uses_readable_format() -> None:
    """session id 使用 YYYYMMDD-HHMMSS-xxxx 格式。"""

    session_id = generate_session_id(datetime(2026, 6, 15, 8, 9, 10))

    assert session_id.startswith("20260615-080910-")
    assert len(session_id) == len("20260615-080910-abcd")


def test_session_store_appends_and_restores_messages(tmp_path: Path) -> None:
    """JSONL 存档可恢复消息，并跳过坏行和孤立工具结果。"""

    store = SessionStore(".agentcode/sessions", project_root=tmp_path)
    session_id = store.create(provider="Fake", model="fake-model")
    store.append_message(session_id, UserMessage(content="你好"))
    store.append_message(
        session_id,
        ToolResultMessage(
            tool_call_id="missing",
            tool_name="read",
            content=text_result("孤立").content,
        ),
    )
    store.append_message(
        session_id,
        AssistantMessage(
            content=[
                TextContent(text="我来读"),
                ToolCall(id="call-1", name="read", arguments={"path": "a.py"}),
            ]
        ),
    )
    store.append_message(
        session_id,
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read",
            content=text_result("内容").content,
        ),
    )
    store.path_for(session_id).write_text(
        store.path_for(session_id).read_text(encoding="utf-8") + "{bad\n",
        encoding="utf-8",
    )

    loaded = store.load(session_id)

    assert loaded.session_id == session_id
    assert [message.role for message in loaded.messages] == [
        "user",
        "assistant",
        "toolResult",
    ]


def test_session_store_cleanup_expired(tmp_path: Path) -> None:
    """过期 session 文件会被启动清理删除。"""

    store = SessionStore(".agentcode/sessions", project_root=tmp_path)
    old_id = store.create(provider="Fake", model="fake-model")
    old_path = store.path_for(old_id)
    old_time = time.time() - 40 * 24 * 60 * 60
    os.utime(old_path, (old_time, old_time))

    assert store.cleanup_expired(30) == 1
    assert not old_path.exists()
