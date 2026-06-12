"""
精确替换文本文件工具。

负责把文件中的唯一旧片段替换为新片段；匹配不唯一时拒绝修改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result


class EditFileTool(BaseTool):
    def name(self) -> str:
        """返回模型调用精确文本替换能力时使用的工具名。"""

        return "edit"

    def description(self) -> str:
        """描述 edit 工具基于唯一旧文本替换的约束。"""

        return "在文本文件中用 new_string 替换唯一匹配的 old_string。"

    def prompt_snippet(self) -> str:
        """返回 system prompt 中的 edit 能力摘要。"""

        return "用精确文本替换修改文件"

    def prompt_guidelines(self) -> list[str]:
        """返回 edit 工具相关的模型行为约束。"""

        return [
            "Use edit for precise changes to existing files.",
            "Read the file first, then keep old_string as small as possible while still unique.",
        ]

    def parameters(self) -> dict[str, object]:
        """声明 edit 工具的路径、旧文本和新文本参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要修改的文件路径"},
                "old_string": {
                    "type": "string",
                    "description": "必须在文件中唯一出现的原文片段",
                },
                "new_string": {"type": "string", "description": "替换后的新文本"},
            },
            "required": ["path", "old_string", "new_string"],
        }

    def execution_mode(self) -> ExecutionMode:
        """edit 会修改文件，必须串行执行以避免写入顺序不确定。"""

        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """按唯一匹配规则修改文件，不唯一或缺失时返回可重试错误。"""

        path_value = args["path"]
        old = args["old_string"]
        new = args["new_string"]
        if old == "":
            return text_result("缺少必填参数: old_string", is_error=True)

        path = Path(path_value)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return text_result(f"读取失败: {exc}", is_error=True)

        count = content.count(old)
        if count == 0:
            return text_result("未找到匹配的内容", is_error=True)
        if count > 1:
            return text_result(
                f"匹配到 {count} 处，old_string 不唯一，请提供更长上下文使其唯一",
                is_error=True,
            )

        try:
            path.write_text(content.replace(old, new, 1), encoding="utf-8")
        except OSError as exc:
            return text_result(f"写入失败: {exc}", is_error=True)
        return text_result(f"已修改 {path}")
