"""
AgentCode 的工具权限判定系统。

负责在工具真正执行前完成黑名单、路径沙箱、规则、权限模式和永久放行配置；
不依赖终端 UI、Provider SDK 或具体工具实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
PermissionVerdict = Literal["allow", "deny", "ask"]
PermissionApproval = Literal["allow_once", "allow_always", "deny_once", "cancel"]
PermissionSource = Literal[
    "blacklist",
    "sandbox",
    "rule",
    "mode",
    "human",
]
ToolCategory = Literal["readonly", "write", "command"]

USER_PERMISSION_PATH = Path.home() / ".agentcode" / "permissions.yaml"
PROJECT_PERMISSION_PATH = Path(".agentcode/permissions.yaml")
LOCAL_PERMISSION_PATH = Path(".agentcode/permissions.local.yaml")
VALID_MODES: tuple[PermissionMode, ...] = (
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
)
MODE_CYCLE: tuple[PermissionMode, ...] = VALID_MODES
READONLY_TOOLS = {"read", "find", "grep", "ls"}
WRITE_TOOLS = {"write", "edit"}
COMMAND_TOOLS = {"bash"}
FRIENDLY_TOOL_NAMES: dict[str, str] = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "find": "Glob",
    "grep": "Grep",
    "ls": "Ls",
}
INTERNAL_TOOL_NAMES = {friendly: internal for internal, friendly in FRIENDLY_TOOL_NAMES.items()}
PLAN_TOOL_NAMES = tuple(sorted(READONLY_TOOLS))
PLAN_REMINDER = (
    "You are in plan mode. Inspect the project with read-only tools only, produce a "
    "clear implementation plan, and do not attempt writes or command execution until "
    "the user exits plan mode."
)

_RULE_RE = re.compile(r"^\s*([A-Za-z]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)
_GLOB_CHARS = set("*?[")
_DANGEROUS_COMMANDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?is)(?:^|[;&|]\s*)rm\s+-[^\n;&|]*[rf][^\n;&|]*[rf][^\n;&|]*\s+"
            r"(?:/|/\*|~|~/|\$HOME\b|\$\{HOME\})"
        ),
        "递归强制删除根目录或家目录",
    ),
    (
        re.compile(
            r"(?is)\bdd\s+[^\n;&|]*\bof=/dev/"
            r"(?:sd[a-z]\d*|hd[a-z]\d*|vd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|disk\d+|rdisk\d+)"
        ),
        "直接写入块设备",
    ),
    (
        re.compile(
            r"(?is)\bmkfs(?:\.[\w-]+)?\s+[^\n;&|]*/dev/"
            r"(?:sd[a-z]\d*|hd[a-z]\d*|vd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|disk\d+|rdisk\d+)"
        ),
        "格式化块设备",
    ),
    (
        re.compile(r"(?is):\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        "fork 炸弹",
    ),
    (
        re.compile(
            r"(?is)(?:>|tee\s+)[^\n;&|]*/dev/"
            r"(?:sd[a-z]\d*|hd[a-z]\d*|vd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|disk\d+|rdisk\d+)"
        ),
        "重定向覆盖块设备",
    ),
    (
        re.compile(
            r"(?is)\b(?:wipefs|shred)\b[^\n;&|]*/dev/"
            r"(?:sd[a-z]\d*|hd[a-z]\d*|vd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|disk\d+|rdisk\d+)"
        ),
        "擦除块设备",
    ),
    (
        re.compile(
            r"(?is)\b(?:chmod|chown)\s+-R\b[^\n;&|]*\s(?:/|~|~/|\$HOME\b|\$\{HOME\})"
        ),
        "递归修改根目录或家目录权限",
    ),
)


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """一条 allow 或 deny 规则，pattern 为空时匹配整个工具。"""

    tool_name: str
    pattern: str | None
    raw: str


@dataclass(frozen=True, slots=True)
class PermissionRuleSet:
    """单个配置层中的权限规则。"""

    allow: tuple[PermissionRule, ...] = ()
    deny: tuple[PermissionRule, ...] = ()


@dataclass(frozen=True, slots=True)
class PermissionLayer:
    """用户级、项目级或本地级配置的解析结果。"""

    name: str
    path: Path
    rules: PermissionRuleSet = field(default_factory=PermissionRuleSet)
    default_mode: PermissionMode | None = None


@dataclass(frozen=True, slots=True)
class PermissionConfig:
    """三层权限配置及启动默认模式。"""

    user: PermissionLayer
    project: PermissionLayer
    local: PermissionLayer
    default_mode: PermissionMode = "default"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """人在回路需要展示和裁决的一次工具调用摘要。"""

    tool_name: str
    friendly_name: str
    arguments: dict[str, Any]
    preview: str
    reason: str
    exact_rule: str


@dataclass(frozen=True, slots=True)
class PermissionCheck:
    """权限流水线的单次判定结果。"""

    verdict: PermissionVerdict
    source: PermissionSource
    reason: str
    request: PermissionRequest | None = None


class PermissionPolicy:
    """封装项目根目录和三层规则，提供单次工具调用判定。"""

    def __init__(self, project_root: str | Path, config: PermissionConfig) -> None:
        """保存已解析项目根和权限配置。"""

        self.project_root = Path(project_root).resolve()
        self.config = config

    @classmethod
    def load(cls, project_root: str | Path) -> "PermissionPolicy":
        """从默认三层配置路径加载权限策略。"""

        root = Path(project_root).resolve()
        return cls(root, load_permission_config(root))

    def default_mode(self) -> PermissionMode:
        """返回配置合并后的启动默认权限模式。"""

        return self.config.default_mode

    def evaluate(
        self,
        tool_name: str,
        args: dict[str, Any],
        mode: PermissionMode,
    ) -> PermissionCheck:
        """按五层流水线判定一次工具调用是否允许执行。"""

        command_block = self._check_blacklist(tool_name, args)
        if command_block is not None:
            return command_block

        sandbox_block = self._check_sandbox(tool_name, args)
        if sandbox_block is not None:
            return sandbox_block

        rule_result = self._check_rules(tool_name, args)
        if rule_result is not None:
            return rule_result

        fallback = mode_fallback(tool_name, mode)
        if fallback == "allow":
            return PermissionCheck("allow", "mode", f"{mode} 模式允许该类工具")

        request = self.build_request(
            tool_name,
            args,
            f"{mode} 模式下该工具调用需要确认",
        )
        return PermissionCheck("ask", "mode", request.reason, request)

    def remember_allow(self, request: PermissionRequest) -> None:
        """把人在回路永久允许的精确规则写入本地级配置并更新内存规则。"""

        local_path = self.config.local.path
        raw = _read_yaml_mapping(local_path) or {}
        allow = raw.get("allow")
        if not isinstance(allow, list):
            allow = []
        if request.exact_rule not in allow:
            allow.append(request.exact_rule)
        raw["allow"] = allow
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.config = load_permission_config(
            self.project_root,
            user_path=self.config.user.path,
            project_path=self.config.project.path,
            local_path=local_path,
        )

    def build_request(
        self,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
    ) -> PermissionRequest:
        """把工具调用转换为 UI 可展示的审批请求。"""

        friendly = friendly_tool_name(tool_name)
        target = rule_match_target(self.project_root, tool_name, args)
        exact = f"{friendly}({target})" if target else friendly
        return PermissionRequest(
            tool_name=tool_name,
            friendly_name=friendly,
            arguments=dict(args),
            preview=target or friendly,
            reason=reason,
            exact_rule=exact,
        )

    def _check_blacklist(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> PermissionCheck | None:
        """命令执行类工具先过不可配置的危险命令黑名单。"""

        if tool_name != "bash":
            return None
        command = args.get("command")
        if not isinstance(command, str):
            return None
        for regex, reason in _DANGEROUS_COMMANDS:
            if regex.search(command):
                return PermissionCheck(
                    "deny",
                    "blacklist",
                    f"危险命令黑名单命中：{reason}",
                )
        return None

    def _check_sandbox(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> PermissionCheck | None:
        """文件类工具限制在项目根目录内，命令执行不做静态路径沙箱。"""

        targets = sandbox_targets(tool_name, args)
        for target in targets:
            if not path_stays_in_project(self.project_root, target):
                return PermissionCheck(
                    "deny",
                    "sandbox",
                    f"路径超出项目根目录：{target}",
                )
        return None

    def _check_rules(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> PermissionCheck | None:
        """按本地、项目、用户三层规则判定，命中即短路。"""

        target = rule_match_target(self.project_root, tool_name, args)
        for layer in (self.config.local, self.config.project, self.config.user):
            if _rules_match(layer.rules.deny, tool_name, target):
                return PermissionCheck(
                    "deny",
                    "rule",
                    f"{layer.name} deny 规则命中",
                )
            if _rules_match(layer.rules.allow, tool_name, target):
                return PermissionCheck(
                    "allow",
                    "rule",
                    f"{layer.name} allow 规则命中",
                )
        return None


def load_permission_config(
    project_root: str | Path,
    *,
    user_path: str | Path | None = None,
    project_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> PermissionConfig:
    """加载三层权限配置，单个文件缺失或格式错误时降级为空规则。"""

    root = Path(project_root).resolve()
    user = _load_layer("用户级", Path(user_path) if user_path else USER_PERMISSION_PATH)
    project = _load_layer(
        "项目级",
        Path(project_path) if project_path else root / PROJECT_PERMISSION_PATH,
    )
    local = _load_layer(
        "本地级",
        Path(local_path) if local_path else root / LOCAL_PERMISSION_PATH,
    )
    mode = _first_mode(local.default_mode, project.default_mode, user.default_mode)
    return PermissionConfig(user=user, project=project, local=local, default_mode=mode)


def friendly_tool_name(tool_name: str) -> str:
    """把内部工具名转换为权限规则使用的友好名。"""

    return FRIENDLY_TOOL_NAMES.get(tool_name, tool_name)


def internal_tool_name(friendly_name: str) -> str | None:
    """把权限规则友好名转换为内部工具名。"""

    return INTERNAL_TOOL_NAMES.get(friendly_name)


def next_permission_mode(mode: PermissionMode) -> PermissionMode:
    """返回 Shift+Tab 循环切换后的下一个权限模式。"""

    index = MODE_CYCLE.index(mode)
    return MODE_CYCLE[(index + 1) % len(MODE_CYCLE)]


def tool_category(tool_name: str) -> ToolCategory | None:
    """返回工具所属权限类别，未知工具不参与权限兜底。"""

    if tool_name in READONLY_TOOLS:
        return "readonly"
    if tool_name in WRITE_TOOLS:
        return "write"
    if tool_name in COMMAND_TOOLS:
        return "command"
    return None


def mode_fallback(tool_name: str, mode: PermissionMode) -> Literal["allow", "ask"]:
    """返回规则未命中时权限模式给出的兜底裁决。"""

    category = tool_category(tool_name)
    if category == "readonly":
        return "allow"
    if mode == "bypassPermissions":
        return "allow"
    if mode == "acceptEdits" and category == "write":
        return "allow"
    return "ask"


def sandbox_targets(tool_name: str, args: dict[str, Any]) -> list[Path]:
    """提取文件类工具需要做沙箱判断的路径表达式。"""

    if tool_name in {"read", "write", "edit"}:
        return _path_arg(args, "path")
    if tool_name == "ls":
        return [Path(_string_arg(args, "path", "."))]
    if tool_name == "find":
        root = _string_arg(args, "path", ".")
        pattern = _string_arg(args, "pattern", ".")
        return [Path(_static_path_prefix(_join_path_pattern(root, pattern)))]
    if tool_name == "grep":
        root = _string_arg(args, "path", ".")
        glob = _string_arg(args, "glob", "**/*")
        return [Path(_static_path_prefix(_join_path_pattern(root, glob)))]
    return []


def path_stays_in_project(project_root: str | Path, target: str | Path) -> bool:
    """解析符号链接后判断目标或最近已存在祖先是否仍在项目内。"""

    root = Path(project_root).resolve()
    raw = Path(target).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    ancestor = _nearest_existing_ancestor(candidate)
    try:
        resolved = ancestor.resolve(strict=True)
    except OSError:
        return False
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    return True


def rule_match_target(
    project_root: str | Path,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """返回规则引擎用于匹配的命令串或项目相对路径。"""

    root = Path(project_root).resolve()
    if tool_name == "bash":
        command = args.get("command")
        return command if isinstance(command, str) else ""
    if tool_name in {"read", "write", "edit", "ls"}:
        default = "." if tool_name == "ls" else ""
        path = _string_arg(args, "path", default)
        return _project_relative_text(root, path)
    if tool_name == "find":
        root_text = _string_arg(args, "path", ".")
        pattern = _string_arg(args, "pattern", ".")
        return _project_relative_text(root, _join_path_pattern(root_text, pattern))
    if tool_name == "grep":
        root_text = _string_arg(args, "path", ".")
        glob = _string_arg(args, "glob", "**/*")
        return _project_relative_text(root, _join_path_pattern(root_text, glob))
    return ""


def denied_tool_result_text(source: PermissionSource, reason: str) -> str:
    """生成回灌给模型的权限拒绝文本。"""

    return f"权限拒绝（{source}）：{reason}"


def _load_layer(name: str, path: Path) -> PermissionLayer:
    """读取单层配置文件，失败时返回空规则层。"""

    raw = _read_yaml_mapping(path)
    if raw is None:
        return PermissionLayer(name=name, path=path)
    return PermissionLayer(
        name=name,
        path=path,
        rules=PermissionRuleSet(
            allow=_parse_rules(raw.get("allow")),
            deny=_parse_rules(raw.get("deny")),
        ),
        default_mode=_parse_mode(raw.get("permission_mode")),
    )


def _read_yaml_mapping(path: Path) -> dict[str, Any] | None:
    """读取 YAML 对象；文件缺失、不可读或格式错误均返回 None。"""

    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def _parse_rules(raw: Any) -> tuple[PermissionRule, ...]:
    """把 YAML 中的规则字符串列表解析为内部规则。"""

    if not isinstance(raw, list):
        return ()
    rules: list[PermissionRule] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        rule = _parse_rule(item)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def _parse_rule(raw: str) -> PermissionRule | None:
    """解析单条 Tool(pattern) 或 Tool 规则。"""

    match = _RULE_RE.match(raw)
    if match is None:
        return None
    friendly = match.group(1)
    tool_name = internal_tool_name(friendly)
    if tool_name is None:
        return None
    pattern = match.group(2)
    return PermissionRule(
        tool_name=tool_name,
        pattern=pattern if pattern is not None else None,
        raw=raw,
    )


def _parse_mode(raw: Any) -> PermissionMode | None:
    """解析配置中的默认权限模式。"""

    if raw in VALID_MODES:
        return cast(PermissionMode, raw)
    return None


def _first_mode(*modes: PermissionMode | None) -> PermissionMode:
    """按配置优先级返回第一个有效模式。"""

    for mode in modes:
        if mode is not None:
            return mode
    return "default"


def _rules_match(
    rules: tuple[PermissionRule, ...],
    tool_name: str,
    target: str,
) -> bool:
    """判断一组规则是否命中当前工具调用。"""

    return any(_rule_matches(rule, tool_name, target) for rule in rules)


def _rule_matches(rule: PermissionRule, tool_name: str, target: str) -> bool:
    """判断单条规则是否命中当前工具调用。"""

    if rule.tool_name != tool_name:
        return False
    if rule.pattern is None:
        return True
    pattern = rule.pattern
    if tool_name == "bash":
        pattern = pattern.replace("**", "*")
    if _has_glob(pattern):
        return fnmatch.fnmatchcase(target, pattern)
    return target == pattern


def _has_glob(pattern: str) -> bool:
    """判断模式是否包含 glob 通配符。"""

    return any(char in pattern for char in _GLOB_CHARS)


def _path_arg(args: dict[str, Any], name: str) -> list[Path]:
    """从参数对象中提取单个路径参数。"""

    value = args.get(name)
    return [Path(value)] if isinstance(value, str) else []


def _string_arg(args: dict[str, Any], name: str, default: str) -> str:
    """从参数对象中读取字符串参数，缺失或类型不对时使用默认值。"""

    value = args.get(name)
    return value if isinstance(value, str) and value else default


def _join_path_pattern(root: str, pattern: str) -> str:
    """按工具语义把搜索根和路径模式拼成一个路径表达式。"""

    pattern_path = PurePosixPath(pattern)
    if pattern_path.is_absolute():
        return pattern
    if root in {"", "."}:
        return pattern
    return str(PurePosixPath(root) / pattern)


def _static_path_prefix(path_text: str) -> str:
    """从含 glob 的路径表达式中取出不含通配符的前缀。"""

    path = PurePosixPath(path_text)
    parts: list[str] = []
    for part in path.parts:
        if any(char in part for char in _GLOB_CHARS):
            break
        parts.append(part)
    if not parts:
        return "."
    return str(PurePosixPath(*parts))


def _project_relative_text(project_root: Path, path_text: str) -> str:
    """把工具路径参数转换为项目相对 POSIX 文本。"""

    path = Path(path_text).expanduser()
    if path.is_absolute():
        try:
            path_text = path.relative_to(project_root).as_posix()
        except ValueError:
            path_text = path.as_posix()
    normalized = posixpath.normpath(path_text.replace("\\", "/"))
    return "." if normalized == "" else normalized


def _nearest_existing_ancestor(path: Path) -> Path:
    """返回路径自身或最近存在的祖先目录。"""

    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current
