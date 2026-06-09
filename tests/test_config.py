from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcode.config import ConfigError, load


class ConfigTests(unittest.TestCase):
    def test_load_valid_config(self) -> None:
        config = load(
            self._write_config("""
providers:
  - name: Claude
    protocol: anthropic
    model: claude-sonnet
    base_url: https://api.anthropic.com
    api_key: sk-ant-test
    thinking: true
  - name: OpenAI
    protocol: openai
    model: gpt-4o
    api_key: sk-openai-test
""")
        )

        self.assertEqual(len(config.providers), 2)
        self.assertEqual(config.providers[0].name, "Claude")
        self.assertEqual(config.providers[0].protocol, "anthropic")
        self.assertTrue(config.providers[0].thinking)
        self.assertIsNone(config.providers[1].base_url)

    def test_missing_provider_field_is_error(self) -> None:
        path = self._write_config("""
providers:
  - name: Claude
    protocol: anthropic
    model: claude-sonnet
""")

        with self.assertRaisesRegex(ConfigError, r"providers\[0\]\.api_key"):
            load(path)

    def test_unknown_protocol_is_error(self) -> None:
        path = self._write_config("""
providers:
  - name: Bad
    protocol: bad
    model: model
    api_key: secret
""")

        with self.assertRaisesRegex(ConfigError, "protocol"):
            load(path)

    def test_missing_file_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ConfigError, "不存在"):
                load(Path(temp_dir) / "missing.yaml")

    def test_top_level_must_be_mapping(self) -> None:
        path = self._write_config("""
- providers
""")

        with self.assertRaisesRegex(ConfigError, "顶层"):
            load(path)

    def test_errors_do_not_include_api_key_value(self) -> None:
        secret = "sk-very-secret"
        path = self._write_config(f"""
providers:
  - name: Bad
    protocol: bad
    model: model
    api_key: {secret}
""")

        with self.assertRaises(ConfigError) as context:
            load(path)

        self.assertNotIn(secret, str(context.exception))

    def _write_config(self, content: str) -> Path:
        temp = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".yaml", delete=False
        )
        with temp:
            temp.write(content.strip() + "\n")
        return Path(temp.name)


if __name__ == "__main__":
    unittest.main()
