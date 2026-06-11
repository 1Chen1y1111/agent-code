"""AgentCode 命令行入口。

负责处理最小启动参数、加载固定配置文件，并把 provider 配置交给 Textual TUI。
"""

from __future__ import annotations

import sys

from agentcode.config import ConfigError, load
from agentcode.tool import new_default_registry
from agentcode.tui import AgentCodeApp

HELP = """Usage: agentcode

启动 AgentCode TUI。

配置文件: .agentcode/config.yaml
"""


def main() -> None:
    # 本期只支持固定 YAML 配置，不提供运行时 flag 覆盖，避免密钥从命令行泄漏。
    if any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
        print(HELP)
        return

    try:
        config = load()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    AgentCodeApp(config.providers, new_default_registry()).run(
        inline=True, inline_no_clear=True
    )
