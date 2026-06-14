"""
MCP 远端工具到 AgentCode Tool 协议的适配层。

负责命名空间、只读 hint 采信、参数 schema 透传，以及工具调用结果转换。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import re
import sys
from typing import Any, Protocol, TextIO

from agentcode.mcp.config import McpServerConfig
from agentcode.permission import ToolCategory
from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result

MCP_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
McpCallTool = Callable[[str, dict[str, Any]], Awaitable[Any]]


class McpSession(Protocol):
    """MCP SDK ClientSession 中本适配层需要的最小调用协议。"""

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """调用远端 MCP tool 并返回 SDK 的 CallToolResult。"""
        ...


class McpTool(BaseTool):
    """把一个远端 MCP tool 包装成 AgentCode 的 Tool 实现。"""

    def __init__(
        self,
        *,
        server_name: str,
        remote_name: str,
        public_name: str,
        description: str,
        input_schema: dict[str, Any],
        read_only: bool,
        session: McpSession,
        stderr: TextIO | None = None,
    ) -> None:
        """保存远端 tool 元数据和调用会话。"""

        self._server_name = server_name
        self._remote_name = remote_name
        self._public_name = public_name
        self._description = description
        self._input_schema = input_schema
        self._read_only = read_only
        self._session = session
        self._stderr = stderr or sys.stderr
        self._warned_non_text_types: set[str] = set()

    def name(self) -> str:
        """返回模型可见的命名空间工具名。"""

        return self._public_name

    def description(self) -> str:
        """返回 provider tools schema 使用的远端工具说明。"""

        return self._description

    def prompt_snippet(self) -> str:
        """返回 system prompt 工具索引中的 MCP 工具摘要。"""

        return self._description

    def parameters(self) -> dict[str, Any]:
        """返回远端 MCP inputSchema 的字典副本。"""

        return dict(self._input_schema)

    def execution_mode(self) -> ExecutionMode:
        """只读 MCP 工具允许并发，其余按可能有副作用串行执行。"""

        return "parallel" if self._read_only else "sequential"

    def permission_category(self) -> ToolCategory | None:
        """返回 MCP 工具供权限兜底使用的动态类别。"""

        return "readonly" if self._read_only else "command"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """调用远端 MCP 工具，并把协议错误转换为可回灌结果。"""

        try:
            result = await self._session.call_tool(self._remote_name, arguments=args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 远端协议错误要回灌给模型。
            return text_result(
                f"MCP 工具 {self._public_name} 调用失败: {exc}",
                is_error=True,
                details=self._details(),
            )
        return self._to_tool_result(result)

    def _to_tool_result(self, result: Any) -> ToolResult:
        """把 SDK CallToolResult 转换为 AgentCode ToolResult。"""

        texts: list[str] = []
        for block in _result_content(result):
            text = _content_text(block)
            if text is not None:
                texts.append(text)
                continue
            self._warn_non_text(_content_type(block))

        structured = _structured_content(result)
        text = "\n".join(texts)
        if not text and structured is not None:
            text = json.dumps(
                structured,
                default=str,
                ensure_ascii=False,
                sort_keys=True,
            )

        details = self._details()
        if structured is not None:
            details["structured_content"] = structured
        return text_result(
            text,
            is_error=_is_error_result(result),
            details=details,
        )

    def _details(self) -> dict[str, Any]:
        """生成工具结果中用于调试和追溯的 MCP 来源信息。"""

        return {
            "mcp_server": self._server_name,
            "mcp_tool": self._remote_name,
        }

    def _warn_non_text(self, content_type: str) -> None:
        """对非 text MCP 内容块按工具和类型告警一次。"""

        if content_type in self._warned_non_text_types:
            return
        self._warned_non_text_types.add(content_type)
        print(
            f"[MCP] 工具 {self._public_name} 返回了非 text 内容块 {content_type}，当前版本已丢弃。",
            file=self._stderr,
        )


def make_mcp_tool(
    config: McpServerConfig,
    remote_tool: Any,
    session: McpSession,
    *,
    stderr: TextIO | None = None,
) -> McpTool | None:
    """把 SDK Tool 对象转换成 McpTool；非法工具名返回 None。"""

    err = stderr or sys.stderr
    remote_name = _remote_name(remote_tool)
    if remote_name is None:
        print(f"[MCP] server {config.name} 返回了缺少 name 的工具，已跳过。", file=err)
        return None

    public_name = mcp_public_tool_name(config.name, remote_name)
    if not MCP_TOOL_NAME_RE.fullmatch(public_name):
        print(
            f"[MCP] 工具名 {public_name} 含有 LLM 工具名不支持的字符，已跳过。",
            file=err,
        )
        return None

    description = _description(remote_tool, config.name)
    input_schema = _input_schema(remote_tool, err, public_name)
    return McpTool(
        server_name=config.name,
        remote_name=remote_name,
        public_name=public_name,
        description=description,
        input_schema=input_schema,
        read_only=_read_only(config, remote_tool),
        session=session,
        stderr=err,
    )


def mcp_public_tool_name(server_name: str, remote_name: str) -> str:
    """生成 MCP 工具对模型暴露的命名空间名称。"""

    return f"mcp__{server_name}__{remote_name}"


def _remote_name(remote_tool: Any) -> str | None:
    """从 SDK Tool 对象读取远端工具名。"""

    name = _attr(remote_tool, "name")
    if isinstance(name, str) and name:
        return name
    return None


def _description(remote_tool: Any, server_name: str) -> str:
    """读取远端工具说明，缺失时生成含 server 名的兜底说明。"""

    description = _attr(remote_tool, "description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return f"MCP tool from server {server_name}."


def _input_schema(
    remote_tool: Any,
    stderr: TextIO,
    public_name: str,
) -> dict[str, Any]:
    """读取远端 inputSchema；异常形态降级为空对象 schema。"""

    schema = _attr(remote_tool, "inputSchema", "input_schema")
    schema_dict = _to_dict(schema)
    if isinstance(schema_dict, dict):
        return dict(schema_dict)
    print(
        f"[MCP] 工具 {public_name} 的 inputSchema 不是对象，已按空参数处理。",
        file=stderr,
    )
    return {"type": "object", "properties": {}}


def _read_only(config: McpServerConfig, remote_tool: Any) -> bool:
    """在 server 被显式信任时采信 annotations.readOnlyHint。"""

    if not config.trust_annotations:
        return False
    annotations = _attr(remote_tool, "annotations")
    read_only = _attr(annotations, "readOnlyHint", "read_only_hint")
    return read_only is True


def _result_content(result: Any) -> list[Any]:
    """读取 CallToolResult.content，缺失或非法时返回空列表。"""

    content = _attr(result, "content")
    return content if isinstance(content, list) else []


def _content_text(block: Any) -> str | None:
    """读取 text 内容块中的文本，非 text 内容返回 None。"""

    if _content_type(block) != "text":
        return None
    text = _attr(block, "text")
    return text if isinstance(text, str) else ""


def _content_type(block: Any) -> str:
    """读取内容块类型，缺失时用对象类型名兜底。"""

    content_type = _attr(block, "type")
    if isinstance(content_type, str) and content_type:
        return content_type
    return type(block).__name__


def _structured_content(result: Any) -> Any | None:
    """读取 CallToolResult.structuredContent。"""

    return _attr(result, "structuredContent", "structured_content")


def _is_error_result(result: Any) -> bool:
    """读取 CallToolResult.isError，缺失时按成功处理。"""

    return _attr(result, "isError", "is_error") is True


def _attr(value: Any, *names: str) -> Any:
    """按多个候选属性名或字典键读取值。"""

    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _to_dict(value: Any) -> dict[str, Any] | None:
    """把 Pydantic 对象或普通 dict 转换成字典。"""

    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(by_alias=True, mode="json", exclude_none=True)
        return dict(dumped) if isinstance(dumped, dict) else None
    return None
