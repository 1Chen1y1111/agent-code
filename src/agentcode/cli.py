"""
AgentCode 命令行入口。

负责处理最小启动参数、加载固定配置文件，并启动普通终端 CLI。
"""

from __future__ import annotations

from pathlib import Path
import sys

from agentcode.config import ConfigError, ContextConfig, load
from agentcode.context import ContextSettings
from agentcode.mcp import load_mcp_server_configs
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

    project_root = Path.cwd()
    prompt_resources = load_prompt_resources(project_root)
    mcp_configs = load_mcp_server_configs(project_root)
    TerminalApp(
        config.providers,
        create_default_registry(),
        prompt_resources.prompt_options,
        mcp_configs=mcp_configs,
        context_settings=_context_settings(config.context),
        project_root=project_root,
        memory_config=config.memory,
    ).run()


def _context_settings(config: ContextConfig) -> ContextSettings:
    """把配置层 context 字段转换为运行期上下文治理设置。"""

    return ContextSettings(
        enabled=config.enabled,
        externalize_tool_results=config.externalize_tool_results,
        max_inline_tool_result_chars=config.max_inline_tool_result_chars,
        max_inline_tool_result_lines=config.max_inline_tool_result_lines,
        tool_result_preview_chars=config.tool_result_preview_chars,
        reserve_tokens=config.reserve_tokens,
        keep_recent_tokens=config.keep_recent_tokens,
        summary_max_tokens=config.summary_max_tokens,
        artifact_root=Path(config.artifact_root) if config.artifact_root else None,
    )
