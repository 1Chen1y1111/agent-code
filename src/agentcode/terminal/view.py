"""普通终端界面的 Rich 渲染片段。

本模块只生成可追加到终端 scrollback 的静态片段，不维护全屏布局或原地替换状态。
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


def assistant_markdown(text: str) -> RenderableType:
    """渲染 assistant 完整正文的 Markdown。"""

    return Markdown(text)


def assistant_text_delta(text: str) -> Text:
    """渲染 assistant 正文增量，避免 Rich markup 误解析模型输出。"""

    return Text(text)


def user_block(text: str) -> Text:
    """渲染用户消息块，便于在 scrollback 中区分输入和回复。"""

    return Text(f"● {text}\n", style="bold cyan")


def thinking_delta(text: str) -> Text:
    """渲染模型 thinking 增量，保持和普通回复正文的视觉区分。"""

    return Text(text, style="dim italic")


def error_block(error: Exception) -> Text:
    """渲染请求失败时的用户可读错误。"""

    return Text(f"● 请求失败：{error}\n", style="bold red")


def elapsed_block(seconds: int) -> Text:
    """渲染本回合最终耗时。"""

    return Text(f"\n耗时：{seconds}s\n\n", style="dim")


def provider_option(index: int, name: str, model: str) -> Text:
    """渲染多 provider 选择列表中的一项。"""

    return Text.assemble(
        (f"{index}. ", "bold cyan"),
        (name, "bold"),
        (" · ", "dim"),
        (model, "dim"),
    )


def tool_start(name: str, args: str) -> Text:
    """渲染工具调用标题，运行中状态由临时 Status 展示。"""

    return Text.assemble(
        ("● ", "bold cyan"),
        (f"{name}({args})\n", "bold"),
    )


def tool_update(result: str, is_error: bool) -> RenderableType:
    """渲染工具执行过程中的增量状态。"""

    return Padding(
        Text(f"↳ {result or '无输出'}\n", style="red" if is_error else "dim"),
        (0, 0, 0, 2),
    )


def tool_result_summary(result: str, is_error: bool) -> RenderableType:
    """渲染工具最终结果摘要，并限制终端中的预览行数。"""

    lines = (result or "无输出").splitlines()
    truncated = len(lines) > 8
    preview = "\n".join(lines[:8])
    if truncated:
        preview = preview.rstrip() + "\n[truncated]"
    return Padding(
        Text(f"⎿ {preview}\n", style="red" if is_error else "dim"),
        (0, 0, 0, 2),
    )
