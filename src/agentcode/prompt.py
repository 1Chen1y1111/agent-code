"""
内置提示词与启动横幅资源。

集中放置 system prompt 和 ASCII banner，避免 UI、provider 适配器各自拼接固定文案。
"""

from __future__ import annotations

from rich.text import Text

SYSTEM_PROMPT = """You are AgentCode, a concise and helpful terminal AI assistant.
You can use tools to read, write, edit files, run shell commands, find files, and search code when needed.
Use tools when you need current project context or need to perform file/command actions.
After tool results are provided, answer concisely and preserve code formatting."""

PET_BANNER = r"""
  /\___/\
 ( -.-  )
<|  ^  |>
  \___/
""".strip("\n")


def render_banner(version: str, cwd: str) -> Text:
    """生成启动时展示的 ASCII banner 和当前工作目录信息。"""

    # 直接返回 Rich Text，避免开启 markup 后误解析用户内容里的方括号。
    banner = Text()
    banner.append(f"{PET_BANNER}\n\n", style="bold yellow")
    banner.append("AgentCode ", style="bold cyan")
    banner.append(f"v{version}\n\n", style="bold white")
    banner.append("cwd: ", style="dim")
    banner.append(f"{cwd}\n\n", style="cyan")
    banner.append("Ready. ", style="bold green")
    banner.append("Tools enabled. No MCP.\n\n", style="dim")
    return banner
