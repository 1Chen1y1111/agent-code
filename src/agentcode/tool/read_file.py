"""
读取文本文件工具。

负责把指定文件内容转成带行号文本；不处理目录递归或二进制语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcode.tool import (
    BaseTool,
    ExecutionMode,
    ToolResult,
    ToolUpdate,
    _truncate,
    text_result,
)

MAX_READ_CHARS = 256 * 1024
MAX_READ_LINES = 2000


class ReadFileTool(BaseTool):
    def name(self) -> str:
        """返回模型调用文件读取能力时使用的工具名。"""

        return "read"

    def description(self) -> str:
        """描述 read 工具读取文本并加行号的输出形态。"""

        return "读取指定文本文件内容，返回带行号的文本。"

    def prompt_snippet(self) -> str:
        """返回 system prompt 中的 read 能力摘要。"""

        return "读取文件内容"

    def prompt_guidelines(self) -> list[str]:
        """返回 read 工具相关的模型行为约束。"""

        return [
            "Use read to examine files instead of cat or sed.",
            "Before editing an existing file, read the relevant content first.",
        ]

    def parameters(self) -> dict[str, object]:
        """声明 read 工具需要的文件路径参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
            },
            "required": ["path"],
        }

    def execution_mode(self) -> ExecutionMode:
        """read 只读取文件内容，可与其他只读工具并发执行。"""

        return "parallel"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """读取指定文本文件，目录、缺失文件和读取错误都转成结果。"""

        path_value = args["path"]

        path = Path(path_value)
        if not path.exists():
            return text_result(f"文件不存在: {path}", is_error=True)
        if path.is_dir():
            return text_result(f"路径是目录，不能读取为文件: {path}", is_error=True)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return text_result(f"读取失败: {exc}", is_error=True)

        numbered = "\n".join(
            f"{line_no:6d}\t{line}"
            for line_no, line in enumerate(text.splitlines(), start=1)
        )
        return text_result(_truncate(numbered, MAX_READ_LINES, MAX_READ_CHARS))
