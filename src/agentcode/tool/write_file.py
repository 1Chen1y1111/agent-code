"""
写入文本文件工具。

负责创建父目录并覆盖写入指定文本；不做权限确认或路径沙箱限制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcode.tool import BaseTool, ExecutionMode, ToolResult, ToolUpdate, text_result


class WriteFileTool(BaseTool):
    def name(self) -> str:
        """返回模型调用文件写入能力时使用的工具名。"""

        return "write"

    def description(self) -> str:
        """描述 write 工具完整覆盖写入文件的行为。"""

        return "创建或覆盖写入指定文本文件，父目录不存在时自动创建。"

    def parameters(self) -> dict[str, object]:
        """声明 write 工具的目标路径和完整内容参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["path", "content"],
        }

    def execution_mode(self) -> ExecutionMode:
        """write 会创建或覆盖文件，必须串行执行以保持结果可预测。"""

        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        on_update: ToolUpdate | None = None,
    ) -> ToolResult:
        """创建父目录后写入完整内容，并把写入失败转为工具错误。"""

        path_value = args["path"]
        content = args["content"]

        path = Path(path_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return text_result(f"写入失败: {exc}", is_error=True)
        return text_result(f"已写入 {path}（{len(content.encode('utf-8'))} 字节）")
