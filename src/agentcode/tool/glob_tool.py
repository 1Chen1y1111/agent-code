"""
文件模式匹配工具。

负责按 glob 模式列出文件路径；只返回文件，不返回目录。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentcode.tool import Result, _load_json_object

MAX_MATCHES = 100


class GlobTool:
    def name(self) -> str:
        """返回模型调用文件查找能力时使用的工具名。"""

        return "find"

    def description(self) -> str:
        """描述 find 工具按 glob 模式返回文件路径的行为。"""

        return "按 glob 模式查找匹配的文件路径。"

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

    async def execute(self, args: str) -> Result:
        """执行文件路径匹配，并限制返回数量避免撑爆上下文。"""

        data, error = _load_json_object(args)
        if error is not None:
            return Result(error, is_error=True)
        pattern = data.get("pattern") if data is not None else None
        root_value = data.get("path", ".") if data is not None else "."
        if not isinstance(pattern, str) or not pattern:
            return Result("缺少必填参数: pattern", is_error=True)
        if not isinstance(root_value, str) or not root_value:
            return Result("参数 path 必须是字符串", is_error=True)

        root = Path(root_value)
        matches: list[str] = []
        try:
            for index, candidate in enumerate(root.glob(pattern), start=1):
                if candidate.is_file():
                    matches.append(str(candidate))
                if index % 100 == 0:
                    await asyncio.sleep(0)
        except OSError as exc:
            return Result(f"查找失败: {exc}", is_error=True)

        if not matches:
            return Result("无匹配")
        matches = sorted(matches)
        suffix = "\n[truncated]" if len(matches) > MAX_MATCHES else ""
        return Result("\n".join(matches[:MAX_MATCHES]) + suffix)
