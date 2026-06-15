from datetime import date

from rich.text import Text

from agentcode.prompt import (
    PET_BANNER,
    PromptBuildOptions,
    PromptContextFile,
    PromptMemoryNote,
    PromptModule,
    SupplementalInstruction,
    build_system_prompt,
    format_supplemental_instruction,
    render_banner,
    runtime_environment_instruction,
)


def test_build_system_prompt_orders_modules_by_priority() -> None:
    prompt = build_system_prompt(
        PromptBuildOptions(
            modules=[
                PromptModule(id="late", priority=20, content="late"),
                PromptModule(id="early", priority=10, content="early"),
            ]
        )
    )

    assert prompt.startswith("early\n\nlate\nCurrent date:")


def test_build_system_prompt_adds_context_and_guidelines() -> None:
    prompt = build_system_prompt(
        PromptBuildOptions(
            selected_tools=["read"],
            tool_snippets={"read": "读取文件"},
            prompt_guidelines=["额外规则"],
            context_files=[PromptContextFile(path="AGENTS.md", content="中文回答")],
        )
    )

    assert "Available tools:" in prompt
    assert "- read: 读取文件" in prompt
    assert "- 额外规则" in prompt
    assert '<project_instructions path="AGENTS.md">' in prompt
    assert "中文回答" in prompt
    assert f"Current date: {date.today().isoformat()}" in prompt


def test_build_system_prompt_adds_memory_notes() -> None:
    prompt = build_system_prompt(
        PromptBuildOptions(
            memory_notes=[
                PromptMemoryNote(
                    category="project",
                    content="使用中文注释",
                    source="20260615-120000-abcd",
                )
            ]
        )
    )

    assert "<agentcode_memory>" in prompt
    assert 'category="project"' in prompt
    assert "使用中文注释" in prompt


def test_runtime_environment_instruction_is_supplemental() -> None:
    instruction = runtime_environment_instruction(
        PromptBuildOptions(cwd="/tmp/project"),
        today=date(2026, 6, 12),
    )

    assert instruction is not None
    rendered = format_supplemental_instruction(instruction)
    assert '<agentcode_supplemental_instruction source="environment"' in rendered
    assert "Do not answer it directly" in rendered
    assert "Current date: 2026-06-12" in rendered
    assert "Current working directory: /tmp/project" in rendered


def test_format_supplemental_instruction_escapes_source_attribute() -> None:
    rendered = format_supplemental_instruction(
        SupplementalInstruction(source='a"b&c', content="remember this")
    )

    assert 'source="a&quot;b&amp;c"' in rendered
    assert "remember this" in rendered


def test_render_banner_returns_styled_text() -> None:
    banner = render_banner("0.1.0", "/tmp/project")

    assert isinstance(banner, Text)
    assert banner.plain.startswith("\n")
    assert PET_BANNER in banner.plain
    assert "AgentCode v0.1.0" in banner.plain
    assert "cwd: /tmp/project" in banner.plain
    assert "Ready. Tools enabled. No MCP." in banner.plain
    assert any(span.style for span in banner.spans)


def test_render_banner_shows_mcp_tool_count() -> None:
    banner = render_banner("0.1.0", "/tmp/project", mcp_tool_count=3)

    assert "Ready. Tools enabled. MCP tools: 3." in banner.plain
    assert "No MCP" not in banner.plain
