"""TUI 中可复用的 Rich 渲染片段。

把用户消息、助手 Markdown、工具状态、错误和状态栏样式集中在这里，避免主应用混入样式细节。
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


def user_block(text: str) -> Text:
    return Text(f"● {text} \n", style="bold cyan")


def assistant_markdown(text: str) -> Markdown:
    # 助手回复结束后再整体 Markdown 渲染，避免流式阶段代码块反复重排。
    return Markdown(text or " ")


def assistant_live(thinking: str, text: str, hide_thinking: bool) -> RenderableType:
    parts: list[RenderableType] = []
    if thinking:
        if hide_thinking:
            parts.append(Text("  Thinking...\n", style="dim italic"))
        else:
            # thinking 用独立样式展示，但不参与最终回答 Markdown 解析。
            parts.append(Text(f"  {thinking.strip()}\n\n", style="dim italic"))
    if text:
        parts.append(assistant_markdown(text))
    if not parts:
        return assistant_markdown("")
    return Group(*parts)


def working_text(frame: str) -> Text:
    return Text.assemble((f"{frame} ", "bold cyan"), ("Working...", "dim"))


def error_block(error: Exception) -> Text:
    return Text(f"● 请求失败：{error} \n", style="bold red")


def elapsed_block(seconds: int) -> Text:
    return Text(f"\n耗时：{seconds}s \n", style="dim")


def status_text(provider_name: str, model: str) -> Text:
    return Text.assemble(
        ("provider: ", "dim"),
        (provider_name, "bold"),
        (" · model: ", "dim"),
        (model, "bold"),
    )


def separator_text(width: int) -> Text:
    return Text("─" * max(width, 1), style="dim")


def tool_line(name: str, args: str) -> Text:
    return Text.assemble(
        ("● ", "bold cyan"),
        (f"{name}({args})\n", "bold"),
    )


def tool_pending(name: str, args: str, seconds: int) -> Text:
    return Text.assemble(
        ("● ", "bold cyan"),
        (f"{name}({args})\n", "bold"),
        (f"  Running... ({seconds}s)\n", "dim"),
    )


def tool_result_summary(result: str, is_error: bool) -> RenderableType:
    lines = (result or "无输出").splitlines()
    truncated = len(lines) > 8
    preview = "\n".join(lines[:8])
    if truncated:
        preview = preview.rstrip() + "\n[truncated]"
    return Padding(
        Text(f"⎿ {preview}\n", style="red" if is_error else "dim"), (0, 0, 0, 2)
    )


def tool_done(name: str, args: str, result: str, is_error: bool) -> RenderableType:
    return Group(tool_line(name, args), tool_result_summary(result, is_error))
