"""
目录列表工具。

负责列出指定目录的直接子项；不递归，不读取文件内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result

MAX_ENTRIES = 200


class LsTool(BaseTool):
    def name(self) -> str:
        """返回模型调用目录列表能力时使用的工具名。"""

        return "ls"

    def description(self) -> str:
        """描述 ls 工具列出目录直接子项的行为。"""

        return "列出指定目录的直接子项，目录以 / 结尾。"

    def parameters(self) -> dict[str, object]:
        """声明 ls 工具的目录路径参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要列出的目录路径，默认当前工作目录",
                },
            },
        }

    def execution_mode(self) -> ExecutionMode:
        """ls 只读取目录结构，可与其他只读工具并发执行。"""

        return "parallel"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """列出目录直接子项，并限制返回数量避免撑爆上下文。"""

        path_value = args.get("path", ".")
        path = Path(path_value)
        if not path.exists():
            return text_result(f"路径不存在: {path}", is_error=True)
        if not path.is_dir():
            return text_result(f"路径不是目录: {path}", is_error=True)

        try:
            entries = [_format_entry(entry) for entry in path.iterdir()]
        except OSError as exc:
            return text_result(f"列目录失败: {exc}", is_error=True)

        if not entries:
            return text_result("目录为空")
        entries = sorted(entries)
        suffix = "\n[truncated]" if len(entries) > MAX_ENTRIES else ""
        return text_result("\n".join(entries[:MAX_ENTRIES]) + suffix)


def _format_entry(path: Path) -> str:
    """把目录项格式化为模型容易区分文件和目录的文本。"""

    suffix = "/" if path.is_dir() else ""
    return f"{path.name}{suffix}"
