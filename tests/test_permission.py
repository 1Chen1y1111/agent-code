from __future__ import annotations

from pathlib import Path

from agentcode.permission import (
    PermissionPolicy,
    load_permission_config,
    mode_fallback,
    next_permission_mode,
    path_stays_in_project,
    rule_match_target,
)


def test_permission_config_merges_layers_by_priority(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    project = tmp_path / "project.yaml"
    local = tmp_path / "local.yaml"
    user.write_text(
        """
permission_mode: acceptEdits
allow:
  - Bash(git status)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    project.write_text(
        """
permission_mode: plan
deny:
  - Bash(git status)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    local.write_text(
        """
permission_mode: bypassPermissions
allow:
  - Bash(git status)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=user,
            project_path=project,
            local_path=local,
        ),
    )

    check = policy.evaluate("bash", {"command": "git status"}, "default")

    assert policy.default_mode() == "bypassPermissions"
    assert check.verdict == "allow"
    assert check.source == "rule"


def test_same_layer_deny_beats_allow(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        """
allow:
  - Bash(git *)
deny:
  - Bash(git status)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=local,
        ),
    )

    check = policy.evaluate("bash", {"command": "git status"}, "default")

    assert check.verdict == "deny"
    assert check.source == "rule"


def test_malformed_config_degrades_to_empty_rules(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text("allow: [", encoding="utf-8")
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=local,
        ),
    )

    check = policy.evaluate("bash", {"command": "git status"}, "default")

    assert check.verdict == "ask"
    assert policy.default_mode() == "default"


def test_blacklist_cannot_be_bypassed(tmp_path: Path) -> None:
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=tmp_path / "missing-local.yaml",
        ),
    )

    check = policy.evaluate("bash", {"command": "rm -rf /"}, "bypassPermissions")

    assert check.verdict == "deny"
    assert check.source == "blacklist"


def test_sandbox_resolves_symlink_before_prefix_check(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-permission-test"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=tmp_path / "missing-local.yaml",
        ),
    )

    check = policy.evaluate("read", {"path": "link/secret.txt"}, "bypassPermissions")

    assert check.verdict == "deny"
    assert check.source == "sandbox"
    assert path_stays_in_project(tmp_path, "new/child.txt")


def test_file_rules_match_project_relative_path_patterns(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        """
allow:
  - Glob(src/**)
  - Grep(src/**/*.py)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    policy = PermissionPolicy(
        tmp_path,
        load_permission_config(
            tmp_path,
            user_path=tmp_path / "missing-user.yaml",
            project_path=tmp_path / "missing-project.yaml",
            local_path=local,
        ),
    )

    glob_check = policy.evaluate(
        "find",
        {"path": "src", "pattern": "**/*.py"},
        "default",
    )
    grep_check = policy.evaluate(
        "grep",
        {"path": "src", "glob": "**/*.py", "pattern": "AgentCode"},
        "default",
    )

    assert glob_check.verdict == "allow"
    assert grep_check.verdict == "allow"
    assert (
        rule_match_target(tmp_path, "grep", {"path": "src", "glob": "**/*.py"})
        == "src/**/*.py"
    )


def test_permission_modes_only_allow_or_ask() -> None:
    assert mode_fallback("read", "default") == "allow"
    assert mode_fallback("write", "default") == "ask"
    assert mode_fallback("write", "acceptEdits") == "allow"
    assert mode_fallback("bash", "acceptEdits") == "ask"
    assert mode_fallback("bash", "bypassPermissions") == "allow"
    assert next_permission_mode("bypassPermissions") == "default"
