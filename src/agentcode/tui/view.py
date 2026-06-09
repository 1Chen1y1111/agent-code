"""TUI 中可复用的 Rich 渲染片段。

把用户消息、助手 Markdown、错误和状态栏样式集中在这里，避免主应用混入样式细节。
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text


def user_block(text: str) -> Text:
    # 完成的用户消息写入 RichLog，和动态流式区分开。
    return Text(f"● {text} \n", style="bold cyan")


def assistant_markdown(text: str) -> Markdown:
    # 助手回复结束后再整体 Markdown 渲染，避免流式阶段代码块反复重排。
    return Markdown(text or " ")


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
