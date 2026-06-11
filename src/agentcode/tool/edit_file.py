"""精确替换文本文件工具。

负责把文件中的唯一旧片段替换为新片段；匹配不唯一时拒绝修改。
"""

from __future__ import annotations

from pathlib import Path

from agentcode.tool import Result, _load_json_object


class EditFileTool:
    def name(self) -> str:
        return "edit_file"

    def description(self) -> str:
        return "在文本文件中用 new_string 替换唯一匹配的 old_string。"

    def parameters(self) -> dict[str, object]:
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

    async def execute(self, args: str) -> Result:
        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        path_value = data.get("path") if data is not None else None
        old = data.get("old_string") if data is not None else None
        new = data.get("new_string") if data is not None else None
        if not isinstance(path_value, str) or not path_value:
            return Result("缺少必填参数: path", is_error=True)
        if not isinstance(old, str) or old == "":
            return Result("缺少必填参数: old_string", is_error=True)
        if not isinstance(new, str):
            return Result("缺少必填参数: new_string", is_error=True)

        path = Path(path_value)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return Result(f"读取失败: {exc}", is_error=True)

        count = content.count(old)
        if count == 0:
            return Result("未找到匹配的内容", is_error=True)
        if count > 1:
            return Result(
                f"匹配到 {count} 处，old_string 不唯一，请提供更长上下文使其唯一",
                is_error=True,
            )

        try:
            path.write_text(content.replace(old, new, 1), encoding="utf-8")
        except OSError as exc:
            return Result(f"写入失败: {exc}", is_error=True)
        return Result(f"已修改 {path}")
