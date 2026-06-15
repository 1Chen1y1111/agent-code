"""
AgentCode 的进程内上下文治理模块。

负责工具结果外部化、token 粗估、会话摘要压缩和上下文溢出识别；不处理
终端 UI、配置文件读取或 provider SDK 细节。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Protocol
from uuid import uuid4

from agentcode.llm import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Provider,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    assistant_tool_calls,
    message_text,
)
from agentcode.tool import ToolResult, content_text, text_result

DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200_000
DEFAULT_OPENAI_CONTEXT_WINDOW = 128_000
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_RESERVE_TOKENS = 16_384
DEFAULT_KEEP_RECENT_TOKENS = 20_000
DEFAULT_MAX_INLINE_TOOL_RESULT_CHARS = 20_000
DEFAULT_MAX_INLINE_TOOL_RESULT_LINES = 400
DEFAULT_TOOL_RESULT_PREVIEW_CHARS = 6_000
DEFAULT_SUMMARY_MAX_TOKENS = 4_096
ESTIMATED_IMAGE_TOKENS = 1_200
TOKEN_CHARS = 4

COMPACTION_SUMMARY_TAG = "agentcode_compaction_summary"
COMPACTION_RECOVERY_TAG = "agentcode_recovery"
SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read the conversation and output "
    "only the requested structured summary. Do not continue the conversation."
)
SUMMARIZATION_PROMPT = """The messages above are conversation history that must be compacted.

Create a structured checkpoint summary that another coding agent can use to continue.

Use this exact Markdown shape:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [User requirements, preferences, and constraints]

## Progress
### Done
- [x] [Completed work]

### In Progress
- [ ] [Current unfinished work]

### Blocked
- [Blockers, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [Next concrete action]

## Critical Context
- [Exact file paths, commands, error messages, or facts needed to continue]

Keep it concise. Preserve exact file paths, symbol names, commands, and errors."""

OVERFLOW_PATTERNS = (
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"request_too_large", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),
    re.compile(r"exceeds (?:the )?context window", re.IGNORECASE),
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit exceeded", re.IGNORECASE),
)

NON_OVERFLOW_PATTERNS = (
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"throttl", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """上下文治理的运行时设置，值由配置文件或默认值提供。"""

    enabled: bool = True
    externalize_tool_results: bool = True
    max_inline_tool_result_chars: int = DEFAULT_MAX_INLINE_TOOL_RESULT_CHARS
    max_inline_tool_result_lines: int = DEFAULT_MAX_INLINE_TOOL_RESULT_LINES
    tool_result_preview_chars: int = DEFAULT_TOOL_RESULT_PREVIEW_CHARS
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS
    artifact_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    """一次上下文 token 粗估结果，拆分真实 usage 和估算增量。"""

    tokens: int
    usage_tokens: int = 0
    trailing_tokens: int = 0
    last_usage_index: int | None = None


@dataclass(frozen=True, slots=True)
class ContextArtifact:
    """一个被外部化的工具结果 artifact 的元数据。"""

    path: str
    original_chars: int
    sha256: str
    preview_chars: int


@dataclass(frozen=True, slots=True)
class CompactionReport:
    """一次压缩完成后对调用方和 UI 可见的摘要信息。"""

    summary: str
    tokens_before: int
    kept_messages: int
    summarized_messages: int
    artifacts: tuple[ContextArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class _CompactionPlan:
    """压缩前选出的摘要范围和保留范围。"""

    messages_to_summarize: list[Message]
    messages_to_keep: list[Message]
    tokens_before: int


class ContextManager:
    """管理单个 AgentSession 生命周期内的上下文压缩状态。"""

    def __init__(
        self,
        settings: ContextSettings | None = None,
        *,
        project_root: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        """保存上下文设置并为 artifact 生成稳定的会话目录。"""

        self.settings = settings or ContextSettings()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.session_id = session_id or uuid4().hex
        self._artifact_dir = self._resolve_artifact_dir()

    @property
    def artifact_dir(self) -> Path:
        """返回当前会话用于保存外部化工具结果的目录。"""

        return self._artifact_dir

    def context_window_for(self, provider: Provider) -> int:
        """根据 provider 暴露信息和协议默认值返回上下文窗口大小。"""

        configured = getattr(provider, "context_window", None)
        if isinstance(configured, int) and configured > 0:
            return configured
        api = provider.api
        if "anthropic" in api:
            return DEFAULT_ANTHROPIC_CONTEXT_WINDOW
        if "openai" in api:
            return DEFAULT_OPENAI_CONTEXT_WINDOW
        return DEFAULT_CONTEXT_WINDOW

    def estimate_context(self, context: Context) -> ContextUsageEstimate:
        """估算完整请求上下文 token，包括 system prompt、消息和工具定义。"""

        base_tokens = estimate_text_tokens(context.system_prompt or "")
        base_tokens += sum(estimate_tool_tokens(tool) for tool in context.tools or [])
        message_estimate = estimate_messages(context.messages)
        return ContextUsageEstimate(
            tokens=base_tokens + message_estimate.tokens,
            usage_tokens=message_estimate.usage_tokens,
            trailing_tokens=base_tokens + message_estimate.trailing_tokens,
            last_usage_index=message_estimate.last_usage_index,
        )

    def should_compact(self, context: Context, provider: Provider) -> bool:
        """判断当前请求是否已经超过自动压缩阈值。"""

        if not self.settings.enabled:
            return False
        estimate = self.estimate_context(context)
        return (
            estimate.tokens
            > self.context_window_for(provider) - self.settings.reserve_tokens
        )

    def externalize_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
    ) -> tuple[ToolResult, ToolResult]:
        """必要时把工具结果落盘，返回 active 结果和 archive 原始结果。"""

        if not self._should_externalize(result):
            return result, result

        original_text = content_text(result.content)
        digest = sha256(original_text.encode("utf-8")).hexdigest()
        filename = f"{_safe_filename(call.id)}_{_safe_filename(call.name)}_{digest[:12]}.txt"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self._artifact_dir / filename
        if not artifact_path.exists():
            artifact_path.write_text(original_text, encoding="utf-8")

        preview = original_text[: self.settings.tool_result_preview_chars]
        if len(preview) < len(original_text):
            preview = preview.rstrip() + "\n[preview truncated]"
        replacement = _externalized_tool_result_text(
            call.name,
            artifact_path,
            original_text,
            digest,
            preview,
        )
        details = {
            **result.details,
            "context_artifact": {
                "externalized": True,
                "path": str(artifact_path),
                "original_chars": len(original_text),
                "sha256": digest,
                "preview_chars": len(preview),
            },
        }
        active = text_result(
            replacement,
            is_error=result.is_error,
            details=details,
            terminate=result.terminate,
        )
        return active, result

    async def compact_conversation(
        self,
        conversation: "ConversationLike",
        provider: Provider,
        tools: Sequence[ToolDefinition],
        *,
        custom_instructions: str | None = None,
    ) -> CompactionReport | None:
        """把旧 active 历史摘要成一条恢复消息，并保留近期消息。"""

        if not self.settings.enabled:
            return None
        messages = conversation.messages()
        plan = self._prepare_compaction(messages)
        if plan is None:
            return None

        summary = await self._generate_summary(
            provider,
            plan.messages_to_summarize,
            custom_instructions=custom_instructions,
        )
        artifacts = tuple(_artifacts_from_messages(plan.messages_to_keep))
        recovery = _recovery_text(plan.messages_to_keep, tools, artifacts)
        compacted_message = UserMessage(
            content=(
                f"<{COMPACTION_SUMMARY_TAG}>\n"
                f"{summary.strip()}\n"
                f"</{COMPACTION_SUMMARY_TAG}>\n\n"
                f"<{COMPACTION_RECOVERY_TAG}>\n"
                f"{recovery.strip()}\n"
                f"</{COMPACTION_RECOVERY_TAG}>"
            )
        )
        conversation.replace_active([compacted_message, *plan.messages_to_keep])
        return CompactionReport(
            summary=summary,
            tokens_before=plan.tokens_before,
            kept_messages=len(plan.messages_to_keep),
            summarized_messages=len(plan.messages_to_summarize),
            artifacts=artifacts,
        )

    def _prepare_compaction(self, messages: list[Message]) -> _CompactionPlan | None:
        """按 keep_recent_tokens 从尾部选出保留消息和待摘要消息。"""

        if len(messages) < 2:
            return None

        tokens_before = estimate_messages(messages).tokens
        keep_start = _find_keep_start(messages, self.settings.keep_recent_tokens)
        if keep_start <= 0:
            return None
        messages_to_summarize = messages[:keep_start]
        messages_to_keep = messages[keep_start:]
        if not messages_to_summarize or not messages_to_keep:
            return None
        return _CompactionPlan(messages_to_summarize, messages_to_keep, tokens_before)

    async def _generate_summary(
        self,
        provider: Provider,
        messages: list[Message],
        *,
        custom_instructions: str | None = None,
    ) -> str:
        """调用当前 provider 生成结构化摘要。"""

        conversation_text = serialize_messages(messages)
        prompt = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
        if custom_instructions and custom_instructions.strip():
            prompt += f"Additional focus: {custom_instructions.strip()}\n\n"
        prompt += SUMMARIZATION_PROMPT
        context = Context(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt)],
            tools=[],
        )
        options = StreamOptions(
            max_tokens=self.settings.summary_max_tokens,
            cache_retention="none",
        )
        return await _complete_text(provider.stream(context, options))

    def _resolve_artifact_dir(self) -> Path:
        """根据设置和 session id 计算 artifact 目录。"""

        if self.settings.artifact_root is None:
            root = self.project_root / ".agentcode" / "context-artifacts"
        else:
            root = self.settings.artifact_root.expanduser()
            if not root.is_absolute():
                root = self.project_root / root
        return root / self.session_id

    def _should_externalize(self, result: ToolResult) -> bool:
        """判断工具结果是否需要落盘外部化。"""

        if not self.settings.enabled or not self.settings.externalize_tool_results:
            return False
        if not result.content:
            return False
        if any(isinstance(block, ImageContent) for block in result.content):
            return False
        text = content_text(result.content)
        if len(text) > self.settings.max_inline_tool_result_chars:
            return True
        return len(text.splitlines()) > self.settings.max_inline_tool_result_lines


class ConversationLike(Protocol):
    """ContextManager 所需的最小 Conversation 协议，避免运行期循环导入。"""

    def messages(self) -> list[Message]:
        """返回当前 active 历史消息。"""

        raise NotImplementedError

    def replace_active(self, messages: list[Message]) -> None:
        """用压缩后的 active 历史替换模型可见消息。"""

        raise NotImplementedError


def estimate_text_tokens(text: str) -> int:
    """用字符数粗估 token 数，保证没有 provider usage 时仍可触发保护。"""

    if not text:
        return 0
    return max(1, (len(text) + TOKEN_CHARS - 1) // TOKEN_CHARS)


def estimate_messages(messages: Sequence[Message]) -> ContextUsageEstimate:
    """估算消息列表 token，优先使用最近一次 assistant usage。"""

    usage_info = _last_usage_info(messages)
    if usage_info is None:
        tokens = sum(estimate_message_tokens(message) for message in messages)
        return ContextUsageEstimate(tokens=tokens, trailing_tokens=tokens)

    usage, index = usage_info
    usage_tokens = _usage_total(usage)
    trailing = sum(estimate_message_tokens(message) for message in messages[index + 1 :])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=index,
    )


def estimate_message_tokens(message: Message) -> int:
    """估算单条内部消息的 token 数。"""

    if isinstance(message, AssistantMessage):
        total = 0
        for block in message.content:
            if isinstance(block, TextContent):
                total += estimate_text_tokens(block.text)
            elif isinstance(block, ToolCall):
                total += estimate_text_tokens(block.name)
                total += estimate_text_tokens(
                    json.dumps(block.arguments, ensure_ascii=False, sort_keys=True)
                )
            else:
                total += estimate_text_tokens(getattr(block, "thinking", ""))
        return max(1, total)
    if isinstance(message, ToolResultMessage):
        return _estimate_content_tokens(message.content)
    return _estimate_user_tokens(message)


def estimate_tool_tokens(tool: ToolDefinition) -> int:
    """估算工具定义在 provider 请求中占用的 token 数。"""

    return estimate_text_tokens(
        json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def serialize_messages(messages: Sequence[Message]) -> str:
    """把消息序列化成摘要模型不会误续写的纯文本。"""

    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            parts.append(f"[user]\n{message_text(message)}")
            continue
        if isinstance(message, AssistantMessage):
            text = message_text(message)
            calls = assistant_tool_calls(message)
            if text:
                parts.append(f"[assistant]\n{text}")
            if calls:
                rendered = "; ".join(
                    f"{call.name}({json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)})"
                    for call in calls
                )
                parts.append(f"[assistant tool calls]\n{rendered}")
            continue
        result_text = message_text(message)
        parts.append(
            f"[tool result: {message.tool_name}]\n"
            f"{_truncate_for_summary(result_text)}"
        )
    return "\n\n".join(parts)


def is_context_overflow(message: AssistantMessage, context_window: int) -> bool:
    """判断 provider 错误或异常 usage 是否代表上下文溢出。"""

    if message.stop_reason == "error" and message.error_message:
        if any(pattern.search(message.error_message) for pattern in NON_OVERFLOW_PATTERNS):
            return False
        return any(pattern.search(message.error_message) for pattern in OVERFLOW_PATTERNS)
    if message.stop_reason == "stop":
        return message.usage.input + message.usage.cache_read > context_window
    if message.stop_reason == "length" and message.usage.output == 0:
        return message.usage.input + message.usage.cache_read >= int(context_window * 0.99)
    return False


async def _complete_text(stream: AsyncIterator[AssistantMessageEvent]) -> str:
    """消费 provider 事件流并返回最终文本。"""

    final: AssistantMessage | None = None
    async for event in stream:
        if isinstance(event, ErrorEvent):
            message = event.error.error_message or "summary failed"
            raise RuntimeError(message)
        if isinstance(event, DoneEvent):
            final = event.message
    if final is None:
        raise RuntimeError("summary failed: missing final message")
    if final.stop_reason == "error":
        raise RuntimeError(final.error_message or "summary failed")
    return message_text(final)


def _find_keep_start(messages: Sequence[Message], keep_recent_tokens: int) -> int:
    """从尾部累计 token，并把切点前移到较安全的消息边界。"""

    accumulated = 0
    index = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[i])
        index = i
        if accumulated >= keep_recent_tokens:
            break
    while index < len(messages) and isinstance(messages[index], ToolResultMessage):
        index += 1
    while index > 0 and not _is_safe_keep_start(messages[index]):
        index -= 1
        if _is_safe_keep_start(messages[index]):
            break
    return max(0, min(index, len(messages)))


def _is_safe_keep_start(message: Message) -> bool:
    """判断 active history 是否适合从这条消息开始继续回放。"""

    return isinstance(message, UserMessage | AssistantMessage)


def _usage_total(usage: Usage) -> int:
    """返回 provider usage 总量，兼容 total_tokens 缺失。"""

    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _last_usage_info(messages: Sequence[Message]) -> tuple[Usage, int] | None:
    """从后往前寻找最近一次可用 assistant usage。"""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            isinstance(message, AssistantMessage)
            and message.stop_reason not in ("aborted", "error")
            and _usage_total(message.usage) > 0
        ):
            return message.usage, index
    return None


def _estimate_user_tokens(message: UserMessage) -> int:
    """估算 user 消息内容 token。"""

    if isinstance(message.content, str):
        return estimate_text_tokens(message.content)
    return _estimate_content_tokens(message.content)


def _estimate_content_tokens(content: Sequence[TextContent | ImageContent]) -> int:
    """估算文本和图片内容块 token。"""

    total = 0
    for block in content:
        if isinstance(block, TextContent):
            total += estimate_text_tokens(block.text)
        elif isinstance(block, ImageContent):
            total += ESTIMATED_IMAGE_TOKENS
    return max(1, total)


def _externalized_tool_result_text(
    tool_name: str,
    artifact_path: Path,
    original_text: str,
    digest: str,
    preview: str,
) -> str:
    """生成进入 active context 的稳定外部化工具结果文本。"""

    return (
        "[agentcode tool result externalized]\n"
        f"tool: {tool_name}\n"
        f"artifact: {artifact_path}\n"
        f"original_chars: {len(original_text)}\n"
        f"sha256: {digest}\n"
        f"preview_chars: {len(preview)}\n\n"
        "<preview>\n"
        f"{preview}\n"
        "</preview>\n\n"
        "Full result was stored on disk. Use read with the artifact path when exact "
        "content is needed."
    )


def _recovery_text(
    kept_messages: Sequence[Message],
    tools: Sequence[ToolDefinition],
    artifacts: Sequence[ContextArtifact],
) -> str:
    """生成摘要后的恢复段，补充模型继续工作需要的边界事实。"""

    read_files, modified_files = _file_operations(kept_messages)
    parts = [
        "Recent original messages after this summary are still included verbatim.",
        "Use available tools when exact omitted content is needed.",
    ]
    if tools:
        parts.append("Available tools: " + ", ".join(tool.name for tool in tools))
    if artifacts:
        parts.append("Externalized tool result artifacts:")
        parts.extend(f"- {artifact.path}" for artifact in artifacts[-10:])
    if read_files:
        parts.append("Recently read files:")
        parts.extend(f"- {path}" for path in read_files[-20:])
    if modified_files:
        parts.append("Recently modified files:")
        parts.extend(f"- {path}" for path in modified_files[-20:])
    return "\n".join(parts)


def _file_operations(messages: Sequence[Message]) -> tuple[list[str], list[str]]:
    """从保留消息的工具调用中提取最近读写文件线索。"""

    read: list[str] = []
    modified: list[str] = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for call in assistant_tool_calls(message):
            path = call.arguments.get("path")
            if not isinstance(path, str):
                continue
            if call.name == "read":
                read.append(path)
            if call.name in {"write", "edit"}:
                modified.append(path)
    return _dedupe(read), _dedupe(modified)


def _artifacts_from_messages(messages: Sequence[Message]) -> list[ContextArtifact]:
    """从保留的 toolResult details 中提取 artifact 元数据。"""

    artifacts: list[ContextArtifact] = []
    for message in messages:
        if not isinstance(message, ToolResultMessage):
            continue
        raw = message.details.get("context_artifact")
        if not isinstance(raw, dict):
            continue
        path = raw.get("path")
        original_chars = raw.get("original_chars")
        digest = raw.get("sha256")
        preview_chars = raw.get("preview_chars")
        if (
            isinstance(path, str)
            and isinstance(original_chars, int)
            and isinstance(digest, str)
            and isinstance(preview_chars, int)
        ):
            artifacts.append(ContextArtifact(path, original_chars, digest, preview_chars))
    return artifacts


def _dedupe(values: Sequence[str]) -> list[str]:
    """按出现顺序去重字符串。"""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _truncate_for_summary(text: str) -> str:
    """限制摘要请求中单个工具结果长度，避免摘要本身撞窗。"""

    max_chars = 2_000
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[summary input truncated]"


def _safe_filename(value: str) -> str:
    """把工具名和调用 id 收窄成可移植文件名片段。"""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "tool"
