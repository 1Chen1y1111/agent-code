"""
AgentCode 命令行入口。

负责处理最小启动参数、加载固定配置文件，并启动普通终端 CLI。
"""

from __future__ import annotations

from pathlib import Path
import sys

from agentcode.config import ConfigError, load
from agentcode.resource_loader import load_prompt_resources
from agentcode.terminal import TerminalApp
from agentcode.tool import create_default_registry

HELP = """Usage: agentcode

启动 AgentCode 终端 CLI。

配置文件: .agentcode/config.yaml
"""


def main() -> None:
    """加载配置并启动 terminal-native 交互循环。"""

    # 本期只支持固定 YAML 配置，不提供运行时 flag 覆盖，避免密钥从命令行泄漏。
    if any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
        print(HELP)
        return

    try:
        config = load()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    prompt_resources = load_prompt_resources(Path.cwd())
    TerminalApp(
        config.providers,
        create_default_registry(),
        prompt_resources.prompt_options,
    ).run()
