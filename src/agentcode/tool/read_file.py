"""
读取文本文件工具。

负责把指定文件内容转成带行号文本；不处理目录递归或二进制语义。
"""

from __future__ import annotations

from pathlib import Path

from agentcode.tool import Result, _load_json_object, _truncate

MAX_READ_CHARS = 256 * 1024
MAX_READ_LINES = 2000


class ReadFileTool:
    def name(self) -> str:
        """返回模型调用文件读取能力时使用的工具名。"""

        return "read"

    def description(self) -> str:
        """描述 read 工具读取文本并加行号的输出形态。"""

        return "读取指定文本文件内容，返回带行号的文本。"

    def parameters(self) -> dict[str, object]:
        """声明 read 工具需要的文件路径参数 schema。"""

        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
            },
            "required": ["path"],
        }

    async def execute(self, args: str) -> Result:
        """读取指定文本文件，目录、缺失文件和读取错误都转成 Result。"""

        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        path_value = data.get("path") if data is not None else None
        if not isinstance(path_value, str) or not path_value:
            return Result("缺少必填参数: path", is_error=True)

        path = Path(path_value)
        if not path.exists():
            return Result(f"文件不存在: {path}", is_error=True)
        if path.is_dir():
            return Result(f"路径是目录，不能读取为文件: {path}", is_error=True)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return Result(f"读取失败: {exc}", is_error=True)

        numbered = "\n".join(
            f"{line_no:6d}\t{line}"
            for line_no, line in enumerate(text.splitlines(), start=1)
        )
        return Result(_truncate(numbered, MAX_READ_LINES, MAX_READ_CHARS))
