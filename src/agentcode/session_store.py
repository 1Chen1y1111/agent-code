"""
AgentCode 的跨进程会话存档模块。

负责生成统一 session id、以 JSONL 追加保存会话消息，并在恢复时容错读取坏行
和修剪孤立工具结果；不处理终端交互或模型调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import secrets
from typing import Any

from agentcode.llm import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserContent,
    UserMessage,
    assistant_tool_calls,
    message_text,
)

SESSION_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredSessionInfo:
    """会话列表中展示和恢复所需的轻量元数据。"""

    session_id: str
    path: Path
    created_at: str
    modified_at: datetime
    provider: str
    model: str
    first_user_message: str


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """从 JSONL 恢复出的会话内容。"""

    session_id: str
    path: Path
    messages: list[Message]
    provider: str
    model: str


def generate_session_id(now: datetime | None = None) -> str:
    """生成统一的可读 session id。"""

    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(2)}"


class SessionStore:
    """管理一个项目下的 JSONL 会话文件。"""

    def __init__(self, root: str | Path, *, project_root: str | Path) -> None:
        """保存会话目录和项目根目录。"""

        self.root = _resolve_under_project(root, project_root)
        self.project_root = Path(project_root).resolve()

    def create(
        self,
        *,
        provider: str,
        model: str,
        session_id: str | None = None,
    ) -> str:
        """创建一个新的 session 文件并写入 header。"""

        self.root.mkdir(parents=True, exist_ok=True)
        resolved_id = session_id or self._unique_session_id()
        path = self.path_for(resolved_id)
        header = {
            "type": "session",
            "version": SESSION_VERSION,
            "id": resolved_id,
            "cwd": str(self.project_root),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
        }
        path.write_text(_json_line(header), encoding="utf-8")
        return resolved_id

    def path_for(self, session_id: str) -> Path:
        """返回某个 session id 对应的 JSONL 路径。"""

        return self.root / f"{session_id}.jsonl"

    def append_message(self, session_id: str, message: Message) -> None:
        """把一条统一消息追加写入 session 文件。"""

        self._append(
            session_id,
            {
                "type": "message",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "message": message_to_json(message),
            },
        )

    def append_compaction(
        self,
        session_id: str,
        *,
        summary: str,
        tokens_before: int,
        kept_messages: int,
        summarized_messages: int,
    ) -> None:
        """记录一次上下文压缩事件，供恢复和审计使用。"""

        self._append(
            session_id,
            {
                "type": "compaction",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
                "tokens_before": tokens_before,
                "kept_messages": kept_messages,
                "summarized_messages": summarized_messages,
            },
        )

    def append_note_update(self, session_id: str, count: int) -> None:
        """记录自动笔记更新数量，方便后续审计。"""

        self._append(
            session_id,
            {
                "type": "note_update",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "count": count,
            },
        )

    def load(self, session_id: str) -> LoadedSession:
        """读取一个 session 文件并恢复可进入对话的消息。"""

        path = self.path_for(session_id)
        header: dict[str, Any] = {}
        messages: list[Message] = []
        for entry in _read_jsonl(path):
            if entry.get("type") == "session":
                header = entry
                continue
            if entry.get("type") != "message":
                continue
            raw_message = entry.get("message")
            if not isinstance(raw_message, dict):
                continue
            message = message_from_json(raw_message)
            if message is not None:
                messages.append(message)

        return LoadedSession(
            session_id=str(header.get("id") or session_id),
            path=path,
            messages=_prune_orphan_tool_results(messages),
            provider=str(header.get("provider") or ""),
            model=str(header.get("model") or ""),
        )

    def list_sessions(self, limit: int | None = None) -> list[StoredSessionInfo]:
        """按最近修改时间倒序列出可恢复会话。"""

        if not self.root.exists():
            return []
        sessions: list[StoredSessionInfo] = []
        for path in self.root.glob("*.jsonl"):
            info = self._session_info(path)
            if info is not None:
                sessions.append(info)
        sessions.sort(key=lambda item: item.modified_at, reverse=True)
        return sessions[:limit] if limit is not None else sessions

    def cleanup_expired(self, retention_days: int) -> int:
        """删除超过保留天数的会话文件并返回删除数量。"""

        if retention_days <= 0 or not self.root.exists():
            return 0
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0
        for path in self.root.glob("*.jsonl"):
            try:
                if datetime.fromtimestamp(path.stat().st_mtime) >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def _append(self, session_id: str, entry: dict[str, Any]) -> None:
        """向 session 文件追加一行 JSON。"""

        self.root.mkdir(parents=True, exist_ok=True)
        with self.path_for(session_id).open("a", encoding="utf-8") as file:
            file.write(_json_line(entry))

    def _unique_session_id(self) -> str:
        """生成当前目录下未冲突的 session id。"""

        for _ in range(100):
            session_id = generate_session_id()
            if not self.path_for(session_id).exists():
                return session_id
        return generate_session_id()

    def _session_info(self, path: Path) -> StoredSessionInfo | None:
        """读取单个 session 文件的列表展示信息。"""

        header: dict[str, Any] | None = None
        first_user = ""
        for entry in _read_jsonl(path):
            if header is None:
                if entry.get("type") != "session":
                    return None
                header = entry
                continue
            if first_user or entry.get("type") != "message":
                continue
            raw_message = entry.get("message")
            if not isinstance(raw_message, dict) or raw_message.get("role") != "user":
                continue
            message = message_from_json(raw_message)
            if message is not None:
                first_user = message_text(message)[:120]

        if header is None:
            return None
        if not first_user:
            return None
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None
        return StoredSessionInfo(
            session_id=str(header.get("id") or path.stem),
            path=path,
            created_at=str(header.get("created_at") or ""),
            modified_at=modified_at,
            provider=str(header.get("provider") or ""),
            model=str(header.get("model") or ""),
            first_user_message=first_user,
        )


def message_to_json(message: Message) -> dict[str, Any]:
    """把统一消息转换为可写入 JSONL 的对象。"""

    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": _user_content_to_json(message.content),
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_assistant_block_to_json(block) for block in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "response_model": message.response_model,
            "response_id": message.response_id,
            "diagnostics": [_diagnostic_to_json(item) for item in message.diagnostics],
            "usage": _usage_to_json(message.usage),
            "stop_reason": message.stop_reason,
            "error_message": message.error_message,
            "timestamp": message.timestamp,
        }
    return {
        "role": "toolResult",
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "content": [_tool_result_block_to_json(block) for block in message.content],
        "details": message.details,
        "is_error": message.is_error,
        "timestamp": message.timestamp,
    }


def message_from_json(raw: dict[str, Any]) -> Message | None:
    """把 JSON 对象恢复为统一消息，无法识别时返回 None。"""

    role = raw.get("role")
    if role == "user":
        return UserMessage(
            content=_user_content_from_json(raw.get("content")),
            timestamp=_int_value(raw.get("timestamp")),
        )
    if role == "assistant":
        return AssistantMessage(
            content=[
                block
                for block in (
                    _assistant_block_from_json(item)
                    for item in _list_value(raw.get("content"))
                )
                if block is not None
            ],
            api=str(raw.get("api") or ""),
            provider=str(raw.get("provider") or ""),
            model=str(raw.get("model") or ""),
            response_model=_optional_str(raw.get("response_model")),
            response_id=_optional_str(raw.get("response_id")),
            diagnostics=[
                item
                for item in (
                    _diagnostic_from_json(value)
                    for value in _list_value(raw.get("diagnostics"))
                )
                if item is not None
            ],
            usage=_usage_from_json(raw.get("usage")),
            stop_reason=raw.get("stop_reason") or "stop",
            error_message=_optional_str(raw.get("error_message")),
            timestamp=_int_value(raw.get("timestamp")),
        )
    if role == "toolResult":
        details = raw.get("details")
        return ToolResultMessage(
            tool_call_id=str(raw.get("tool_call_id") or ""),
            tool_name=str(raw.get("tool_name") or ""),
            content=[
                block
                for block in (
                    _tool_result_block_from_json(item)
                    for item in _list_value(raw.get("content"))
                )
                if block is not None
            ],
            details=details if isinstance(details, dict) else {},
            is_error=bool(raw.get("is_error")),
            timestamp=_int_value(raw.get("timestamp")),
        )
    return None


def _resolve_under_project(path: str | Path, project_root: str | Path) -> Path:
    """把相对路径解析到项目根目录下，绝对路径保持原样。"""

    root = Path(project_root).resolve()
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """容错读取 JSONL 文件，坏行会被跳过。"""

    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return entries
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def _json_line(value: dict[str, Any]) -> str:
    """把对象序列化为一行稳定 JSON。"""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _prune_orphan_tool_results(messages: list[Message]) -> list[Message]:
    """恢复时移除没有对应 toolCall 的 toolResult。"""

    valid_tool_call_ids: set[str] = set()
    restored: list[Message] = []
    for message in messages:
        if isinstance(message, AssistantMessage):
            valid_tool_call_ids.update(call.id for call in assistant_tool_calls(message))
            restored.append(message)
            continue
        if isinstance(message, ToolResultMessage):
            if message.tool_call_id in valid_tool_call_ids:
                restored.append(message)
            continue
        restored.append(message)
    return restored


def _user_content_to_json(content: object) -> object:
    """序列化 user content。"""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return [_text_or_image_to_json(block) for block in content]


def _user_content_from_json(raw: object) -> UserContent:
    """恢复 user content。"""

    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    return [
        block
        for block in (_text_or_image_from_json(item) for item in raw)
        if block is not None
    ]


def _assistant_block_to_json(block: AssistantContent) -> dict[str, Any]:
    """序列化 assistant content block。"""

    if isinstance(block, TextContent):
        return {
            "type": "text",
            "text": block.text,
            "text_signature": block.text_signature,
        }
    if isinstance(block, ThinkingContent):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    return {
        "type": "toolCall",
        "id": block.id,
        "name": block.name,
        "arguments": block.arguments,
        "thought_signature": block.thought_signature,
    }


def _assistant_block_from_json(raw: object) -> AssistantContent | None:
    """恢复 assistant content block。"""

    if not isinstance(raw, dict):
        return None
    block_type = raw.get("type")
    if block_type == "text":
        return TextContent(
            text=str(raw.get("text") or ""),
            text_signature=_optional_str(raw.get("text_signature")),
        )
    if block_type == "thinking":
        return ThinkingContent(
            thinking=str(raw.get("thinking") or ""),
            thinking_signature=_optional_str(raw.get("thinking_signature")),
            redacted=bool(raw.get("redacted")),
        )
    if block_type == "toolCall":
        arguments = raw.get("arguments")
        return ToolCall(
            id=str(raw.get("id") or ""),
            name=str(raw.get("name") or ""),
            arguments=arguments if isinstance(arguments, dict) else {},
            thought_signature=_optional_str(raw.get("thought_signature")),
        )
    return None


def _tool_result_block_to_json(block: ToolResultContent) -> dict[str, Any]:
    """序列化 toolResult content block。"""

    return _text_or_image_to_json(block)


def _tool_result_block_from_json(raw: object) -> ToolResultContent | None:
    """恢复 toolResult content block。"""

    return _text_or_image_from_json(raw)


def _text_or_image_to_json(block: TextContent | ImageContent) -> dict[str, Any]:
    """序列化 text 或 image content block。"""

    if isinstance(block, TextContent):
        return {
            "type": "text",
            "text": block.text,
            "text_signature": block.text_signature,
        }
    return {
        "type": "image",
        "data": block.data,
        "mime_type": block.mime_type,
    }


def _text_or_image_from_json(raw: object) -> TextContent | ImageContent | None:
    """恢复 text 或 image content block。"""

    if not isinstance(raw, dict):
        return None
    if raw.get("type") == "text":
        return TextContent(
            text=str(raw.get("text") or ""),
            text_signature=_optional_str(raw.get("text_signature")),
        )
    if raw.get("type") == "image":
        return ImageContent(
            data=str(raw.get("data") or ""),
            mime_type=str(raw.get("mime_type") or ""),
        )
    return None


def _usage_to_json(usage: Usage) -> dict[str, Any]:
    """序列化 token 用量。"""

    return {
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "total_tokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cache_read": usage.cost.cache_read,
            "cache_write": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }


def _usage_from_json(raw: object) -> Usage:
    """恢复 token 用量。"""

    if not isinstance(raw, dict):
        return Usage()
    cost = raw.get("cost")
    cost_data = cost if isinstance(cost, dict) else {}
    return Usage(
        input=_int_value(raw.get("input")),
        output=_int_value(raw.get("output")),
        cache_read=_int_value(raw.get("cache_read")),
        cache_write=_int_value(raw.get("cache_write")),
        total_tokens=_int_value(raw.get("total_tokens")),
        cost=UsageCost(
            input=_float_value(cost_data.get("input")),
            output=_float_value(cost_data.get("output")),
            cache_read=_float_value(cost_data.get("cache_read")),
            cache_write=_float_value(cost_data.get("cache_write")),
            total=_float_value(cost_data.get("total")),
        ),
    )


def _diagnostic_to_json(diagnostic: AssistantMessageDiagnostic) -> dict[str, Any]:
    """序列化 provider 诊断信息。"""

    return {
        "type": diagnostic.type,
        "timestamp": diagnostic.timestamp,
        "error": (
            {
                "message": diagnostic.error.message,
                "name": diagnostic.error.name,
                "stack": diagnostic.error.stack,
                "code": diagnostic.error.code,
            }
            if diagnostic.error is not None
            else None
        ),
        "details": diagnostic.details,
    }


def _diagnostic_from_json(raw: object) -> AssistantMessageDiagnostic | None:
    """恢复 provider 诊断信息。"""

    if not isinstance(raw, dict):
        return None
    error = raw.get("error")
    error_info = None
    if isinstance(error, dict):
        error_info = DiagnosticErrorInfo(
            message=str(error.get("message") or ""),
            name=_optional_str(error.get("name")),
            stack=_optional_str(error.get("stack")),
            code=error.get("code"),
        )
    details = raw.get("details")
    return AssistantMessageDiagnostic(
        type=str(raw.get("type") or ""),
        timestamp=_int_value(raw.get("timestamp")),
        error=error_info,
        details=details if isinstance(details, dict) else {},
    )


def _list_value(raw: object) -> list[object]:
    """把未知值收窄为列表。"""

    return raw if isinstance(raw, list) else []


def _optional_str(raw: object) -> str | None:
    """把未知值收窄为可选字符串。"""

    return raw if isinstance(raw, str) else None


def _int_value(raw: object) -> int:
    """把未知值安全转换为整数。"""

    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


def _float_value(raw: object) -> float:
    """把未知值安全转换为浮点数。"""

    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    return 0.0
