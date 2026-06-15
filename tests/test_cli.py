from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from agentcode.cli import main
from agentcode.config import Config, ConfigError, ProviderConfig
from agentcode.context import ContextSettings
from agentcode.prompt import PromptBuildOptions


class CliTests(unittest.TestCase):
    """覆盖命令行入口的参数、配置错误和应用启动路径。"""

    def test_help_prints_without_starting_app(self) -> None:
        """`--help` 只打印帮助文本，不加载配置或启动应用。"""

        output = io.StringIO()

        with (
            patch("sys.argv", ["agentcode", "--help"]),
            patch("agentcode.cli.load") as load,
            redirect_stdout(output),
        ):
            main()

        load.assert_not_called()
        self.assertIn("Usage: agentcode", output.getvalue())
        self.assertIn(".agentcode/config.yaml", output.getvalue())

    def test_starts_terminal_app(self) -> None:
        """正常启动时把配置、工具和提示资源交给 TerminalApp。"""

        app = Mock()
        config = Config(
            providers=[
                ProviderConfig(
                    name="Only",
                    protocol="openai",
                    api_key="test-key",
                    model="test-model",
                )
            ]
        )
        prompt_resources = Mock(prompt_options=PromptBuildOptions())

        with (
            patch("sys.argv", ["agentcode"]),
            patch("agentcode.cli.load", return_value=config),
            patch("agentcode.cli.load_prompt_resources", return_value=prompt_resources),
            patch("agentcode.cli.load_mcp_server_configs", return_value=("mcp",)),
            patch("agentcode.cli.create_default_registry", return_value="registry"),
            patch("agentcode.cli.TerminalApp", return_value=app) as app_cls,
        ):
            main()

        app_cls.assert_called_once_with(
            config.providers,
            "registry",
            prompt_resources.prompt_options,
            mcp_configs=("mcp",),
            context_settings=ContextSettings(),
            project_root=Path.cwd(),
            memory_config=config.memory,
        )
        app.run.assert_called_once_with()

    def test_config_error_exits_with_stderr_message(self) -> None:
        """配置错误会打印用户可读信息并以退出码 1 结束。"""

        err = io.StringIO()

        with (
            patch("sys.argv", ["agentcode"]),
            patch("agentcode.cli.load", side_effect=ConfigError("bad config")),
            redirect_stderr(err),
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("配置错误：bad config", err.getvalue())


if __name__ == "__main__":
    unittest.main()
