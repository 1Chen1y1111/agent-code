from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from agentcode.mcp.config import McpServerConfig
from agentcode.mcp.tool import make_mcp_tool
from agentcode.tool import content_text


@pytest.mark.asyncio
async def test_mcp_tool_maps_metadata_and_readonly_hint() -> None:
    """可信 server 的 readOnlyHint 会映射成只读权限类别和并发执行。"""

    session = FakeSession(
        SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    )
    tool = make_mcp_tool(
        McpServerConfig("github", "http", url="http://example", trust_annotations=True),
        _remote_tool(
            name="get_issue",
            description="Get an issue",
            annotations=SimpleNamespace(readOnlyHint=True),
        ),
        session,
    )

    assert tool is not None
    assert tool.name() == "mcp__github__get_issue"
    assert tool.description() == "Get an issue"
    assert tool.prompt_snippet() == "Get an issue"
    assert tool.parameters()["required"] == ["id"]
    assert tool.execution_mode() == "parallel"
    assert tool.permission_category() == "readonly"

    result = await tool.execute("call_1", {"id": 1})

    assert content_text(result.content) == "ok"
    assert session.calls == [("get_issue", {"id": 1})]


def test_mcp_tool_does_not_trust_readonly_hint_by_default() -> None:
    """未显式信任 annotations 时，readOnlyHint 不影响安全分类。"""

    tool = make_mcp_tool(
        McpServerConfig("github", "http", url="http://example"),
        _remote_tool(
            name="get_issue",
            description="Get an issue",
            annotations=SimpleNamespace(readOnlyHint=True),
        ),
        FakeSession(SimpleNamespace(content=[])),
    )

    assert tool is not None
    assert tool.execution_mode() == "sequential"
    assert tool.permission_category() == "command"


@pytest.mark.asyncio
async def test_mcp_tool_maps_structured_and_non_text_results() -> None:
    """MCP 调用结果保留 structuredContent，并对非 text 内容告警一次。"""

    stderr = io.StringIO()
    session = FakeSession(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="visible"),
                SimpleNamespace(type="image", data="..."),
                SimpleNamespace(type="image", data="..."),
            ],
            structuredContent={"ok": True},
            isError=True,
        )
    )
    tool = make_mcp_tool(
        McpServerConfig("demo", "stdio", command="demo"),
        _remote_tool(name="fetch", description="Fetch"),
        session,
        stderr=stderr,
    )

    assert tool is not None
    result = await tool.execute("call_1", {})

    assert result.is_error is True
    assert content_text(result.content) == "visible"
    assert result.details["structured_content"] == {"ok": True}
    assert stderr.getvalue().count("非 text 内容块 image") == 1


@pytest.mark.asyncio
async def test_mcp_tool_serializes_structured_result_when_text_is_empty() -> None:
    """没有 text 内容时，structuredContent 会序列化后回灌给模型。"""

    session = FakeSession(SimpleNamespace(content=[], structuredContent={"answer": 42}))
    tool = make_mcp_tool(
        McpServerConfig("demo", "stdio", command="demo"),
        _remote_tool(name="json_tool", description="JSON"),
        session,
    )

    assert tool is not None
    result = await tool.execute("call_1", {})

    assert content_text(result.content) == '{"answer": 42}'
    assert result.details["structured_content"] == {"answer": 42}


@pytest.mark.asyncio
async def test_mcp_tool_protocol_error_is_tool_error() -> None:
    """远端协议异常会转成 is_error 工具结果，而不是抛出到 Agent Loop。"""

    tool = make_mcp_tool(
        McpServerConfig("demo", "stdio", command="demo"),
        _remote_tool(name="boom", description="Boom"),
        RaisingSession(),
    )

    assert tool is not None
    result = await tool.execute("call_1", {})

    assert result.is_error is True
    assert "调用失败" in content_text(result.content)


def test_mcp_tool_rejects_invalid_public_name() -> None:
    """命名空间拼接后含非法字符的 MCP 工具会被跳过。"""

    stderr = io.StringIO()

    tool = make_mcp_tool(
        McpServerConfig("bad.server", "stdio", command="demo"),
        _remote_tool(name="ok", description="OK"),
        FakeSession(SimpleNamespace(content=[])),
        stderr=stderr,
    )

    assert tool is None
    assert "不支持的字符" in stderr.getvalue()


def _remote_tool(
    *,
    name: str,
    description: str,
    annotations: object | None = None,
) -> SimpleNamespace:
    """创建最小远端 MCP Tool 形态。"""

    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
        annotations=annotations,
    )


class FakeSession:
    """记录 MCP call_tool 参数并返回固定结果的测试会话。"""

    def __init__(self, result: Any) -> None:
        """保存固定返回值。"""

        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """记录调用并返回固定结果。"""

        self.calls.append((name, dict(arguments or {})))
        return self.result


class RaisingSession:
    """调用时抛异常的测试会话。"""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """模拟远端协议或传输失败。"""

        raise RuntimeError("remote down")
