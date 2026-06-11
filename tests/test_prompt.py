from rich.text import Text

from agentcode.prompt import PET_BANNER, render_banner


def test_render_banner_returns_styled_text() -> None:
    banner = render_banner("0.1.0", "/tmp/project")

    assert isinstance(banner, Text)
    assert PET_BANNER in banner.plain
    assert "AgentCode v0.1.0" in banner.plain
    assert "cwd: /tmp/project" in banner.plain
    assert "Ready. Tools enabled. No MCP." in banner.plain
    assert any(span.style for span in banner.spans)
