"""
命令执行工具。

负责异步执行 shell 命令并返回 stdout/stderr/退出码；超时由注册中心统一控制。
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentcode.tool import (
    BaseTool,
    ExecutionMode,
    ToolResult,
    ToolUpdate,
    _truncate,
    text_result,
)


class BashTool(BaseTool):
    def name(self) -> str:
        """返回模型调用命令执行能力时使用的工具名。"""

        return "bash"

    def description(self) -> str:
        """描述 shell 执行工具的能力和输出形态。"""

        return "在当前工作目录执行 shell 命令，返回 stdout、stderr 和退出码。"

    def parameters(self) -> dict[str, object]:
        """声明 bash 工具接受的 JSON 参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
            },
            "required": ["command"],
        }

    def execution_mode(self) -> ExecutionMode:
        """bash 可能产生副作用，必须在同一批工具调用中串行执行。"""

        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """执行 shell 命令，并把进程输出和退出码汇总给模型。"""

        command = args["command"]

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await proc.communicate()
        except asyncio.CancelledError:
            # 注册中心用 wait_for 控制超时；协程被取消时必须同步终止子进程，
            # 否则用户已经看到超时结果，后台命令却仍可能继续运行。
            proc.kill()
            await proc.wait()
            raise
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        content = f"exit_code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        return text_result(_truncate(content, max_lines=10000, max_chars=30000))
