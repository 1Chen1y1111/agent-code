"""
TUI 中可复用的 Rich 渲染片段。

把用户消息、助手 Markdown、工具状态、错误和状态栏样式集中在这里，避免主应用混入样式细节。
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


def user_block(text: str) -> Text:
    """渲染用户消息块。"""

    return Text(f"● {text} \n", style="bold cyan")


def assistant_final(thinking: str, text: str, hide_thinking: bool) -> RenderableType:
    """把助手最终回复渲染为 Markdown。"""

    parts = _thinking_parts(thinking, hide_thinking)
    if text:
        if not parts:
            return Markdown(text)
        parts.append(Markdown(text))
    if not parts:
        return Markdown(" ")
    return Group(*parts)


def assistant_streaming(
    thinking: str, text: str, hide_thinking: bool
) -> RenderableType:
    """渲染流式中的助手消息，回复正文先按纯文本展示。"""

    parts = _thinking_parts(thinking, hide_thinking)
    if text:
        if not parts:
            return Text(text)
        parts.append(Text(text))
    if not parts:
        return Text(" ")
    return Group(*parts)


def _thinking_parts(thinking: str, hide_thinking: bool) -> list[RenderableType]:
    """生成 thinking 展示片段，保证它不进入 Markdown 解析。"""

    if not thinking:
        return []
    if hide_thinking:
        return [Text("Thinking...\n", style="dim italic")]
    return [Text(f"{thinking.strip()}\n", style="dim italic")]


def working_text(frame: str) -> Text:
    """渲染聊天流末尾的工作中 spinner 文本。"""

    return Text.assemble((f" \n{frame} ", "bold cyan"), ("Working...", "dim"))


def error_block(error: Exception) -> Text:
    """渲染请求失败时插入聊天流的错误消息。"""

    return Text(f"● 请求失败：{error} \n", style="bold red")


def elapsed_block(seconds: int) -> Text:
    """渲染本回合耗时信息。"""

    return Text(f"\n耗时：{seconds}s \n", style="dim")


def status_text(provider_name: str, model: str) -> Text:
    """渲染底部状态栏中的 provider 和 model 信息。"""

    return Text.assemble(
        ("provider: ", "dim"),
        (provider_name, "bold"),
        (" · model: ", "dim"),
        (model, "bold"),
    )


def separator_text(width: int) -> Text:
    """生成指定宽度的浅色分隔线。"""

    return Text("─" * max(width, 1), style="dim")


def tool_line(name: str, args: str) -> Text:
    """渲染工具调用标题行。"""

    return Text.assemble(
        ("● ", "bold cyan"),
        (f"{name}({args})\n", "bold"),
    )


def tool_pending(name: str, args: str, seconds: int) -> Text:
    """渲染正在执行中的工具调用状态。"""

    return Text.assemble(
        ("● ", "bold cyan"),
        (f"{name}({args})\n", "bold"),
        (f"  Running... ({seconds}s)\n", "dim"),
    )


def tool_result_summary(result: str, is_error: bool) -> RenderableType:
    """渲染工具结果摘要，并限制聊天里展示的行数。"""

    lines = (result or "无输出").splitlines()
    truncated = len(lines) > 8
    preview = "\n".join(lines[:8])
    if truncated:
        preview = preview.rstrip() + "\n[truncated]"
    return Padding(
        Text(f"⎿ {preview}\n", style="red" if is_error else "dim"), (0, 0, 0, 2)
    )


def tool_done(name: str, args: str, result: str, is_error: bool) -> RenderableType:
    """渲染已完成的工具调用和结果摘要。"""

    return Group(tool_line(name, args), tool_result_summary(result, is_error))
