"""
读取并校验 AgentCode 的 YAML 配置。

本模块只负责启动期配置加载，避免把配置来源、密钥处理和 provider 校验分散到 UI 或 SDK 适配器中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DEFAULT_CONFIG_PATH = Path(".agentcode/config.yaml")
Protocol = Literal["anthropic", "openai"]


class ConfigError(Exception):
    """用户可读的配置错误。"""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    # 这些字段直接对应 YAML 中的 provider 配置；密钥只用于 SDK 初始化，不进入 UI 渲染。
    name: str
    protocol: Protocol
    api_key: str
    model: str
    base_url: str | None = None
    thinking: bool = False
    context_window: int | None = None


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """上下文治理配置；省略时使用跨 provider 的保守默认值。"""

    enabled: bool = True
    externalize_tool_results: bool = True
    max_inline_tool_result_chars: int = 20_000
    max_inline_tool_result_lines: int = 400
    tool_result_preview_chars: int = 6_000
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000
    summary_max_tokens: int = 4_096
    artifact_root: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    providers: list[ProviderConfig] = field(default_factory=list)
    context: ContextConfig = field(default_factory=ContextConfig)


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """读取 YAML 配置文件并返回启动期可用的结构化配置。"""

    # 配置错误需要在启动期变成可读信息，因此这里统一转换为 ConfigError。
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 格式无效: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError("配置文件顶层必须是 YAML 对象")

    providers_raw = raw_config.get("providers")
    if not isinstance(providers_raw, list) or not providers_raw:
        raise ConfigError("providers 必须是非空列表")

    providers = [
        _parse_provider(provider, index) for index, provider in enumerate(providers_raw)
    ]
    return Config(providers=providers, context=_parse_context(raw_config.get("context")))


def _parse_provider(raw_provider: Any, index: int) -> ProviderConfig:
    """校验单个 provider 配置，并把普通字符串收窄成内部协议类型。"""

    # prefix 会进入错误信息，方便用户定位是哪一项 provider 配错。
    prefix = f"providers[{index}]"
    if not isinstance(raw_provider, dict):
        raise ConfigError(f"{prefix} 必须是 YAML 对象")

    protocol_raw = _required_string(raw_provider, "protocol", prefix)
    if protocol_raw not in ("anthropic", "openai"):
        raise ConfigError(f"{prefix}.protocol 只支持 anthropic 或 openai")
    # 运行时校验完成后收窄为 Literal，避免把字符串类型噪声扩散到调用方。
    protocol = cast(Protocol, protocol_raw)

    base_url = raw_provider.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ConfigError(f"{prefix}.base_url 必须是非空字符串或省略")

    thinking = raw_provider.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ConfigError(f"{prefix}.thinking 必须是布尔值")

    context_window = _optional_positive_int(raw_provider, "context_window", prefix)

    return ProviderConfig(
        name=_required_string(raw_provider, "name", prefix),
        protocol=protocol,
        api_key=_required_string(raw_provider, "api_key", prefix),
        model=_required_string(raw_provider, "model", prefix),
        base_url=base_url.strip() if isinstance(base_url, str) else None,
        thinking=thinking,
        context_window=context_window,
    )


def _parse_context(raw_context: Any) -> ContextConfig:
    """校验顶层 context 配置，并填补默认值。"""

    if raw_context is None:
        return ContextConfig()
    if not isinstance(raw_context, dict):
        raise ConfigError("context 必须是 YAML 对象或省略")

    prefix = "context"
    enabled = _optional_bool(raw_context, "enabled", prefix, True)
    externalize = _optional_bool(
        raw_context,
        "externalize_tool_results",
        prefix,
        True,
    )
    artifact_root = raw_context.get("artifact_root")
    if artifact_root is not None and (
        not isinstance(artifact_root, str) or not artifact_root.strip()
    ):
        raise ConfigError(f"{prefix}.artifact_root 必须是非空字符串或省略")

    return ContextConfig(
        enabled=enabled,
        externalize_tool_results=externalize,
        max_inline_tool_result_chars=_positive_int(
            raw_context,
            "max_inline_tool_result_chars",
            prefix,
            20_000,
        ),
        max_inline_tool_result_lines=_positive_int(
            raw_context,
            "max_inline_tool_result_lines",
            prefix,
            400,
        ),
        tool_result_preview_chars=_positive_int(
            raw_context,
            "tool_result_preview_chars",
            prefix,
            6_000,
        ),
        reserve_tokens=_positive_int(raw_context, "reserve_tokens", prefix, 16_384),
        keep_recent_tokens=_positive_int(
            raw_context,
            "keep_recent_tokens",
            prefix,
            20_000,
        ),
        summary_max_tokens=_positive_int(
            raw_context,
            "summary_max_tokens",
            prefix,
            4_096,
        ),
        artifact_root=artifact_root.strip() if isinstance(artifact_root, str) else None,
    )


def _required_string(raw_provider: dict[str, Any], field_name: str, prefix: str) -> str:
    """读取必填字符串字段，并统一处理空值报错。"""

    value = raw_provider.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{prefix}.{field_name} 不能为空")
    return value.strip()


def _optional_bool(
    raw: dict[str, Any],
    field_name: str,
    prefix: str,
    default: bool,
) -> bool:
    """读取可选布尔字段，缺省时返回默认值。"""

    value = raw.get(field_name, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{prefix}.{field_name} 必须是布尔值")
    return value


def _positive_int(
    raw: dict[str, Any],
    field_name: str,
    prefix: str,
    default: int,
) -> int:
    """读取可选正整数字段，并应用默认值。"""

    value = raw.get(field_name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{prefix}.{field_name} 必须是正整数")
    return value


def _optional_positive_int(
    raw: dict[str, Any],
    field_name: str,
    prefix: str,
) -> int | None:
    """读取可选正整数字段，省略时返回 None。"""

    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{prefix}.{field_name} 必须是正整数或省略")
    return value
