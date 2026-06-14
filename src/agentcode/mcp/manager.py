"""
AgentCode 的 MCP server 生命周期管理。

负责启动多个 MCP client 会话、发现工具并注册到 Registry，退出时集中关闭连接。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
import os
import sys
from typing import Any, TextIO

import httpx
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from agentcode.mcp.config import McpServerConfig
from agentcode.mcp.tool import McpSession, make_mcp_tool
from agentcode.tool import Registry

MCP_OPERATION_TIMEOUT = 30.0
MCP_SHUTDOWN_TIMEOUT = 5.0


@dataclass(slots=True)
class McpConnection:
    """一个已初始化 MCP server 会话及其关闭栈。"""

    server_name: str
    session: McpSession
    tools: list[Any]
    exit_stack: AsyncExitStack

    async def close(self) -> None:
        """关闭该 server 的 MCP session 和底层传输。"""

        await self.exit_stack.aclose()


class McpManager:
    """管理本进程内所有 MCP server 连接和工具注册。"""

    def __init__(
        self,
        configs: Sequence[McpServerConfig],
        registry: Registry,
        *,
        stderr: TextIO | None = None,
    ) -> None:
        """保存启动配置、工具注册中心和告警输出。"""

        self._configs = tuple(configs)
        self._registry = registry
        self._stderr = stderr or sys.stderr
        self._connections: list[McpConnection] = []
        self.tool_count = 0

    async def start(self) -> int:
        """并发连接所有配置的 MCP server，并返回成功注册的工具数量。"""

        if not self._configs:
            return 0
        counts = await asyncio.gather(
            *(self._start_one(config) for config in self._configs),
        )
        self.tool_count = sum(counts)
        return self.tool_count

    async def close(self) -> None:
        """在整体 5 秒兜底内关闭所有已建立 MCP 连接。"""

        if not self._connections:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(connection.close() for connection in self._connections),
                    return_exceptions=True,
                ),
                timeout=MCP_SHUTDOWN_TIMEOUT,
            )
        except TimeoutError:
            _warn(self._stderr, "MCP 连接关闭超过 5s，已放弃等待。")
        finally:
            self._connections.clear()

    async def _start_one(self, config: McpServerConfig) -> int:
        """启动单个 server，失败时告警并返回 0。"""

        try:
            connection = await asyncio.wait_for(
                _open_connection(config),
                timeout=MCP_OPERATION_TIMEOUT,
            )
        except TimeoutError:
            _warn(self._stderr, f"MCP server {config.name} 启动超过 30s，已跳过。")
            return 0
        except Exception as exc:  # noqa: BLE001 - 单 server 失败不能影响启动。
            _warn(self._stderr, f"MCP server {config.name} 启动失败，已跳过：{exc}")
            return 0

        count = self._register_tools(config, connection)
        if count == 0:
            await connection.close()
            return 0
        self._connections.append(connection)
        return count

    def _register_tools(
        self,
        config: McpServerConfig,
        connection: McpConnection,
    ) -> int:
        """把一个 server 的远端 tools 包装并注册进 Registry。"""

        count = 0
        for remote_tool in connection.tools:
            tool = make_mcp_tool(
                config,
                remote_tool,
                connection.session,
                stderr=self._stderr,
            )
            if tool is None:
                continue
            try:
                self._registry.register(tool)
            except ValueError:
                _warn(self._stderr, f"MCP 工具 {tool.name()} 重名，已保留先注册者。")
                continue
            count += 1
        return count


async def _open_connection(config: McpServerConfig) -> McpConnection:
    """建立一个 MCP 连接并完成 initialize 和 tools/list。"""

    stack = AsyncExitStack()
    try:
        read, write = await _open_transport(config, stack)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools_list = await _list_all_tools(session)
        return McpConnection(
            server_name=config.name,
            session=session,
            tools=tools_list,
            exit_stack=stack,
        )
    except BaseException:
        await stack.aclose()
        raise


async def _open_transport(
    config: McpServerConfig,
    stack: AsyncExitStack,
) -> tuple[Any, Any]:
    """按 server 类型打开 stdio 或 Streamable HTTP 传输。"""

    if config.type == "stdio":
        env = dict(os.environ)
        env.update(config.env)
        params = StdioServerParameters(
            command=config.command or "",
            args=list(config.args),
            env=env,
        )
        return await stack.enter_async_context(stdio_client(params))

    client = await stack.enter_async_context(httpx.AsyncClient(headers=config.headers))
    transport = await stack.enter_async_context(
        streamable_http_client(config.url or "", http_client=client)
    )
    return transport[0], transport[1]


async def _list_all_tools(session: ClientSession) -> list[Any]:
    """分页读取远端 server 暴露的全部 tools。"""

    tools_list: list[Any] = []
    cursor: str | None = None
    while True:
        if cursor:
            response = await session.list_tools(
                params=types.PaginatedRequestParams(cursor=cursor)
            )
        else:
            response = await session.list_tools()
        tools_list.extend(response.tools)
        cursor = _next_cursor(response)
        if not cursor:
            return tools_list


def _next_cursor(response: Any) -> str | None:
    """从 SDK list result 中读取下一页 cursor。"""

    cursor = getattr(response, "nextCursor", None)
    if isinstance(cursor, str) and cursor:
        return cursor
    cursor = getattr(response, "next_cursor", None)
    return cursor if isinstance(cursor, str) and cursor else None


def _warn(stderr: TextIO, message: str) -> None:
    """向 stderr 输出统一前缀的 MCP 生命周期告警。"""

    print(f"[MCP] {message}", file=stderr)
