from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from agentcode.cli import main
from agentcode.config import Config, ProviderConfig


class CliTests(unittest.TestCase):
    def test_help_prints_without_starting_app(self) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["agentcode", "--help"]), redirect_stdout(output):
            main()

        self.assertIn("Usage: agentcode", output.getvalue())
        self.assertIn(".agentcode/config.yaml", output.getvalue())

    def test_starts_textual_inline_app(self) -> None:
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

        with (
            patch("sys.argv", ["agentcode"]),
            patch("agentcode.cli.load", return_value=config),
            patch("agentcode.cli.new_default_registry", return_value="registry"),
            patch("agentcode.cli.AgentCodeApp", return_value=app) as app_cls,
        ):
            main()

        app_cls.assert_called_once_with(config.providers, "registry")
        app.run.assert_called_once_with(inline=True, inline_no_clear=True)


if __name__ == "__main__":
    unittest.main()
