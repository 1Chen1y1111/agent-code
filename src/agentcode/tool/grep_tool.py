"""
文本内容搜索工具。

负责用 Python 正则搜索文件内容并返回文件、行号和命中行。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from agentcode.tool import Result, _load_json_object

MAX_MATCHES = 100
MAX_LINE_CHARS = 1_000_000


class GrepTool:
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

    async def execute(self, args: str) -> Result:
        """搜索匹配内容，跳过不可读文件并截断过大的命中集合。"""

        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        pattern = data.get("pattern") if data is not None else None
        root_value = data.get("path", ".") if data is not None else "."
        glob_value = data.get("glob") if data is not None else None
        if not isinstance(pattern, str) or not pattern:
            return Result("缺少必填参数: pattern", is_error=True)
        if not isinstance(root_value, str) or not root_value:
            return Result("参数 path 必须是字符串", is_error=True)
        if glob_value is not None and not isinstance(glob_value, str):
            return Result("参数 glob 必须是字符串", is_error=True)

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return Result(f"正则非法: {exc}", is_error=True)

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
                            return Result("\n".join(matches) + "\n[truncated]")
            except (OSError, UnicodeDecodeError):
                continue
            await asyncio.sleep(0)

        if not matches:
            return Result("无命中")
        return Result("\n".join(matches))
