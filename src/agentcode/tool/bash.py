"""命令执行工具。

负责异步执行 shell 命令并返回 stdout/stderr/退出码；超时由注册中心统一控制。
"""

from __future__ import annotations

import asyncio

from agentcode.tool import Result, _load_json_object, _truncate


class BashTool:
    def name(self) -> str:
        return "bash"

    def description(self) -> str:
        return "在当前工作目录执行 shell 命令，返回 stdout、stderr 和退出码。"

    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
            },
            "required": ["command"],
        }

    async def execute(self, args: str) -> Result:
        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        command = data.get("command") if data is not None else None
        if not isinstance(command, str) or not command:
            return Result("缺少必填参数: command", is_error=True)

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
        return Result(_truncate(content, max_lines=10000, max_chars=30000))
