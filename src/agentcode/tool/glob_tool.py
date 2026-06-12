"""
文件模式匹配工具。

负责按 glob 模式列出文件路径；只返回文件，不返回目录。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result

MAX_MATCHES = 100


class GlobTool(BaseTool):
    def name(self) -> str:
        """返回模型调用文件查找能力时使用的工具名。"""

        return "find"

    def description(self) -> str:
        """描述 find 工具按 glob 模式返回文件路径的行为。"""

        return "按 glob 模式查找匹配的文件路径。"

    def prompt_snippet(self) -> str:
        """返回 system prompt 中的 find 能力摘要。"""

        return "按 glob 查找文件路径"

    def prompt_guidelines(self) -> list[str]:
        """返回 find 工具相关的模型行为约束。"""

        return ["Use find to locate files by path pattern instead of bash."]

    def parameters(self) -> dict[str, object]:
        """声明 find 工具的匹配模式和可选根目录 schema。"""

        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 模式，例如 **/*.py",
                },
                "path": {
                    "type": "string",
                    "description": "可选根目录，默认当前工作目录",
                },
            },
            "required": ["pattern"],
        }

    def execution_mode(self) -> ExecutionMode:
        """find 只读取目录结构，可与其他只读工具并发执行。"""

        return "parallel"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """执行文件路径匹配，并限制返回数量避免撑爆上下文。"""

        pattern = args["pattern"]
        root_value = args.get("path", ".")

        root = Path(root_value)
        matches: list[str] = []
        try:
            for index, candidate in enumerate(root.glob(pattern), start=1):
                if candidate.is_file():
                    matches.append(str(candidate))
                if index % 100 == 0:
                    await asyncio.sleep(0)
        except OSError as exc:
            return text_result(f"查找失败: {exc}", is_error=True)

        if not matches:
            return text_result("无匹配")
        matches = sorted(matches)
        suffix = "\n[truncated]" if len(matches) > MAX_MATCHES else ""
        return text_result("\n".join(matches[:MAX_MATCHES]) + suffix)
