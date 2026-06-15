"""
AgentCode 的长期记忆笔记模块。

负责把自然结束后的对话增量提取为持久化笔记，并在后续请求中按关键词召回；
不负责会话 JSONL 存档或终端交互。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

from agentcode.context import serialize_messages
from agentcode.llm import (
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Provider,
    StreamOptions,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    UserMessage,
    message_text,
)
from agentcode.prompt import PromptMemoryNote

NOTE_CATEGORIES = {"user", "project", "work"}
NOTE_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable memory notes for a coding agent. "
    "Return only JSON and do not continue the conversation."
)
NOTE_EXTRACTION_PROMPT = """Extract durable notes worth remembering for future coding sessions.

Return a JSON array. Each item must have:
- category: one of "user", "project", "work"
- content: one short concrete note
- confidence: number from 0 to 1

Remember stable user preferences, project conventions, unresolved work, exact paths,
or decisions that would help future sessions. Do not store transient tool output,
generic facts, or information that is only useful for the immediate next token.

If nothing is worth remembering, return [].
"""


@dataclass(frozen=True, slots=True)
class MemoryNote:
    """一条持久化记忆笔记。"""

    category: str
    content: str
    confidence: float
    source_session_id: str
    created_at: str


class MemoryStore:
    """管理用户级和项目级笔记文件。"""

    def __init__(
        self,
        notes_dir: str | Path,
        *,
        project_root: str | Path,
        user_notes_dir: str | Path | None = None,
    ) -> None:
        """保存笔记目录，user 类笔记默认写入用户级目录。"""

        self.project_root = Path(project_root).resolve()
        self.notes_dir = _resolve_under_project(notes_dir, self.project_root)
        self.user_notes_dir = (
            Path(user_notes_dir).expanduser().resolve()
            if user_notes_dir is not None
            else Path.home() / ".agentcode" / "memory"
        )

    def save_notes(self, notes: Sequence[MemoryNote]) -> int:
        """保存去重后的笔记并返回实际新增数量。"""

        existing = {self._dedupe_key(note.content) for note in self.load_all()}
        added = 0
        for note in notes:
            content = note.content.strip()
            if not content:
                continue
            key = self._dedupe_key(content)
            if key in existing:
                continue
            existing.add(key)
            path = self._path_for(note.category)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(_json_line(note_to_json(note)))
            added += 1
        return added

    def load_all(self) -> list[MemoryNote]:
        """读取所有分类笔记，坏行会被跳过。"""

        notes: list[MemoryNote] = []
        for path in self._note_paths():
            notes.extend(_read_notes(path))
        return notes

    def relevant_notes(self, query: str, *, limit: int = 10) -> list[PromptMemoryNote]:
        """按关键词召回与当前输入相关的笔记。"""

        notes = self.load_all()
        if not notes:
            return []
        query_terms = _terms(query)
        scored = [
            (_score_note(note, query_terms), note.created_at, note)
            for note in notes
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [note for score, _, note in scored if score > 0]
        if not selected:
            selected = [note for _, _, note in scored]
        return [
            PromptMemoryNote(
                category=note.category,
                content=note.content,
                source=note.source_session_id,
            )
            for note in selected[:limit]
        ]

    def _path_for(self, category: str) -> Path:
        """返回分类笔记文件路径。"""

        normalized = category if category in NOTE_CATEGORIES else "work"
        if normalized == "user":
            return self.user_notes_dir / "user.jsonl"
        return self.notes_dir / f"{normalized}.jsonl"

    def _note_paths(self) -> list[Path]:
        """返回所有可能存在的笔记文件路径。"""

        return [
            self.user_notes_dir / "user.jsonl",
            self.notes_dir / "project.jsonl",
            self.notes_dir / "work.jsonl",
        ]

    def _dedupe_key(self, content: str) -> str:
        """生成笔记去重键。"""

        return " ".join(content.casefold().split())


class MemoryExtractor:
    """用当前 provider 从回合增量中提取持久化笔记。"""

    def __init__(self, *, max_tokens: int = 1_000) -> None:
        """保存提取请求的最大输出 token 数。"""

        self.max_tokens = max_tokens

    async def extract(
        self,
        provider: Provider,
        messages: Sequence[Message],
        *,
        session_id: str,
    ) -> list[MemoryNote]:
        """调用 LLM 提取笔记，失败或无内容时返回空列表。"""

        if not messages:
            return []
        prompt = (
            "<conversation_delta>\n"
            f"{serialize_messages(messages)}\n"
            "</conversation_delta>\n\n"
            f"{NOTE_EXTRACTION_PROMPT}"
        )
        context = Context(
            system_prompt=NOTE_EXTRACTION_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt)],
            tools=[],
        )
        try:
            text = await _complete_text(
                provider.stream(
                    context,
                    StreamOptions(max_tokens=self.max_tokens, cache_retention="none"),
                )
            )
        except Exception:
            return []
        return _parse_notes(text, session_id=session_id)


def note_to_json(note: MemoryNote) -> dict[str, object]:
    """把笔记转换为 JSONL 对象。"""

    return {
        "category": note.category,
        "content": note.content,
        "confidence": note.confidence,
        "source_session_id": note.source_session_id,
        "created_at": note.created_at,
    }


def note_from_json(raw: dict[str, object]) -> MemoryNote | None:
    """从 JSON 对象恢复笔记。"""

    content = raw.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    category = raw.get("category")
    confidence = raw.get("confidence")
    source = raw.get("source_session_id")
    created_at = raw.get("created_at")
    return MemoryNote(
        category=category if isinstance(category, str) else "work",
        content=content.strip(),
        confidence=_float_value(confidence),
        source_session_id=source if isinstance(source, str) else "",
        created_at=created_at if isinstance(created_at, str) else "",
    )


def _parse_notes(text: str, *, session_id: str) -> list[MemoryNote]:
    """解析模型返回的 JSON 笔记数组。"""

    payload = _json_payload(text)
    if payload is None:
        return []
    raw_notes: object
    try:
        raw_notes = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(raw_notes, dict):
        raw_notes = raw_notes.get("notes", [])
    if not isinstance(raw_notes, list):
        return []

    created_at = datetime.now().isoformat(timespec="seconds")
    notes: list[MemoryNote] = []
    for raw in raw_notes:
        if not isinstance(raw, dict):
            continue
        category = raw.get("category")
        content = raw.get("content")
        confidence = _float_value(raw.get("confidence"))
        if not isinstance(content, str) or not content.strip():
            continue
        normalized_category = category if isinstance(category, str) else "work"
        if normalized_category not in NOTE_CATEGORIES:
            normalized_category = "work"
        if confidence <= 0:
            confidence = 0.5
        notes.append(
            MemoryNote(
                category=normalized_category,
                content=content.strip(),
                confidence=confidence,
                source_session_id=session_id,
                created_at=created_at,
            )
        )
    return notes


def _json_payload(text: str) -> str | None:
    """从模型文本中抽取 JSON 数组或对象。"""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped
    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if 0 <= array_start < array_end:
        return stripped[array_start : array_end + 1]
    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if 0 <= object_start < object_end:
        return stripped[object_start : object_end + 1]
    return None


async def _complete_text(stream: AsyncIterator[AssistantMessageEvent]) -> str:
    """收集一次简单 completion 的文本输出。"""

    chunks: list[str] = []
    async for event in stream:
        if isinstance(event, TextDeltaEvent):
            chunks.append(event.delta)
        elif isinstance(event, TextEndEvent):
            chunks.append(event.content)
        elif isinstance(event, DoneEvent):
            text = message_text(event.message)
            if text:
                return text
        elif isinstance(event, ErrorEvent):
            return message_text(event.error)
        elif isinstance(event, TextStartEvent):
            continue
    return "".join(chunks)


def _read_notes(path: Path) -> list[MemoryNote]:
    """容错读取一个笔记 JSONL 文件。"""

    notes: list[MemoryNote] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return notes
    for line in lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        note = note_from_json(raw)
        if note is not None:
            notes.append(note)
    return notes


def _resolve_under_project(path: str | Path, project_root: Path) -> Path:
    """把相对路径解析到项目根目录下，绝对路径保持原样。"""

    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (project_root / value).resolve()


def _json_line(value: dict[str, object]) -> str:
    """把对象序列化为一行 JSON。"""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _terms(text: str) -> set[str]:
    """把查询文本切分为粗粒度关键词集合。"""

    return {
        term
        for term in re.findall(r"[\w\-/\\.]+", text.casefold())
        if len(term) >= 2
    }


def _score_note(note: MemoryNote, query_terms: set[str]) -> int:
    """计算笔记和查询的简单关键词匹配分数。"""

    if not query_terms:
        return 0
    note_terms = _terms(note.content)
    return len(query_terms & note_terms)


def _float_value(raw: object) -> float:
    """把未知值安全转换为浮点数。"""

    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    return 0.0
