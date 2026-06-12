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


def assistant_markdown(text: str) -> Markdown:
    """把助手最终回复渲染为 Markdown。"""

    # 助手回复结束后再整体 Markdown 渲染，避免流式阶段代码块反复重排。
    return Markdown(text or " ")


def assistant_live(thinking: str, text: str, hide_thinking: bool) -> RenderableType:
    """渲染流式中的助手消息，包括可隐藏的 thinking 通道。"""

    parts: list[RenderableType] = []
    if thinking:
        if hide_thinking:
            parts.append(Text("  Thinking...\n", style="dim italic"))
        else:
            # thinking 用独立样式展示，但不参与最终回答 Markdown 解析。
            parts.append(Text(f"  {thinking.strip()}\n", style="dim italic"))
    if text:
        parts.append(assistant_markdown(text))
    if not parts:
        return assistant_markdown("")
    return Group(*parts)


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
