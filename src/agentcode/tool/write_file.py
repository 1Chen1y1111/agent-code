"""写入文本文件工具。

负责创建父目录并覆盖写入指定文本；不做权限确认或路径沙箱限制。
"""

from __future__ import annotations

from pathlib import Path

from agentcode.tool import Result, _load_json_object


class WriteFileTool:
    def name(self) -> str:
        return "write_file"

    def description(self) -> str:
        return "创建或覆盖写入指定文本文件，父目录不存在时自动创建。"

    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, args: str) -> Result:
        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        path_value = data.get("path") if data is not None else None
        content = data.get("content") if data is not None else None
        if not isinstance(path_value, str) or not path_value:
            return Result("缺少必填参数: path", is_error=True)
        if not isinstance(content, str):
            return Result("缺少必填参数: content", is_error=True)

        path = Path(path_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return Result(f"写入失败: {exc}", is_error=True)
        return Result(f"已写入 {path}（{len(content.encode('utf-8'))} 字节）")
