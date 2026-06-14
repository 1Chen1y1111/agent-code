from __future__ import annotations

from contextlib import AsyncExitStack
import io
from types import SimpleNamespace
from typing import Any

import pytest

from agentcode.mcp.config import McpServerConfig
from agentcode.mcp.manager import McpConnection, McpManager
from agentcode.tool import Registry, content_text


@pytest.mark.asyncio
async def test_mcp_manager_registers_tools_and_keeps_successful_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manager 会注册成功 server 的工具，并跳过重复工具名。"""

    registry = Registry()
    stderr = io.StringIO()
    connection = McpConnection(
        server_name="demo",
        session=FakeSession(),
        tools=[
            _remote_tool("one"),
            _remote_tool("one"),
            _remote_tool("two"),
        ],
        exit_stack=AsyncExitStack(),
    )

    async def open_connection(config: McpServerConfig) -> McpConnection:
        """返回固定连接，避免测试真实 MCP 传输。"""

        return connection

    monkeypatch.setattr("agentcode.mcp.manager._open_connection", open_connection)
    manager = McpManager(
        [McpServerConfig("demo", "stdio", command="demo")],
        registry,
        stderr=stderr,
    )

    count = await manager.start()
    result = await registry.execute("call_1", "mcp__demo__one", {}, timeout=1)
    await manager.close()

    assert count == 2
    assert registry.names() == ("mcp__demo__one", "mcp__demo__two")
    assert content_text(result.content) == "ok"
    assert "重名" in stderr.getvalue()


@pytest.mark.asyncio
async def test_mcp_manager_isolates_failed_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单个 MCP server 启动失败只告警并跳过自身。"""

    registry = Registry()
    stderr = io.StringIO()

    async def open_connection(config: McpServerConfig) -> McpConnection:
        """根据 server 名模拟成功或失败连接。"""

        if config.name == "bad":
            raise RuntimeError("boom")
        return McpConnection(
            server_name=config.name,
            session=FakeSession(),
            tools=[_remote_tool("tool")],
            exit_stack=AsyncExitStack(),
        )

    monkeypatch.setattr("agentcode.mcp.manager._open_connection", open_connection)
    manager = McpManager(
        [
            McpServerConfig("bad", "stdio", command="bad"),
            McpServerConfig("good", "stdio", command="good"),
        ],
        registry,
        stderr=stderr,
    )

    count = await manager.start()
    await manager.close()

    assert count == 1
    assert registry.names() == ("mcp__good__tool",)
    assert "bad 启动失败" in stderr.getvalue()


def _remote_tool(name: str) -> SimpleNamespace:
    """创建最小远端工具定义。"""

    return SimpleNamespace(
        name=name,
        description=name,
        inputSchema={"type": "object", "properties": {}},
    )


class FakeSession:
    """返回固定文本结果的 MCP 测试会话。"""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """模拟远端工具调用。"""

        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
