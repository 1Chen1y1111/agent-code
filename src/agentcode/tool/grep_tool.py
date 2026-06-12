"""
文本内容搜索工具。

负责用 Python 正则搜索文件内容并返回文件、行号和命中行。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result

MAX_MATCHES = 100
MAX_LINE_CHARS = 1_000_000


class GrepTool(BaseTool):
    def name(self) -> str:
        """返回模型调用内容搜索能力时使用的工具名。"""

        return "grep"

    def description(self) -> str:
        """描述 grep 工具返回 file:line:content 命中列表的行为。"""

        return "用 Python 正则搜索文件内容，返回 file:line:content 命中列表。"

    def parameters(self) -> dict[str, object]:
        """声明 grep 工具的正则、根目录和文件过滤参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python 正则表达式"},
                "path": {
                    "type": "string",
                    "description": "可选搜索根目录，默认当前工作目录",
                },
                "glob": {
                    "type": "string",
                    "description": "可选文件名 glob 过滤，例如 **/*.py",
                },
            },
            "required": ["pattern"],
        }

    def execution_mode(self) -> ExecutionMode:
        """grep 只读取文件内容，可与其他只读工具并发执行。"""

        return "parallel"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """搜索匹配内容，跳过不可读文件并截断过大的命中集合。"""

        pattern = args["pattern"]
        root_value = args.get("path", ".")
        glob_value = args.get("glob")

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return text_result(f"正则非法: {exc}", is_error=True)

        root = Path(root_value)
        matches: list[str] = []
        files = root.rglob(glob_value or "*")
        for file in files:
            if not file.is_file():
                continue
            try:
                with file.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if len(line) > MAX_LINE_CHARS:
                            matches.append(f"{file}:{line_no}:该行过长，未完整搜索")
                            break
                        if regex.search(line):
                            matches.append(f"{file}:{line_no}:{line.rstrip()}")
                        if len(matches) >= MAX_MATCHES:
                            return text_result("\n".join(matches) + "\n[truncated]")
            except (OSError, UnicodeDecodeError):
                continue
            await asyncio.sleep(0)

        if not matches:
            return text_result("无命中")
        return text_result("\n".join(matches))
