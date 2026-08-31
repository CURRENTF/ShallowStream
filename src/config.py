"""Shared JSON config loading helpers."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_repo_path(path: str) -> str:
    path = str(path).strip()
    if not path:
        raise ValueError("config path must not be empty")
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root(), path))


def load_json_object(path: str) -> Dict[str, Any]:
    resolved = resolve_repo_path(path)
    with open(resolved, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved}")
    return payload


def parse_json_object(raw: str, label: str) -> Dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def merge_known_keys(
    config: Dict[str, Any],
    payload: Mapping[str, Any],
    *,
    name: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    aliases = aliases or {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        key = aliases.get(key, key)
        normalized[key] = value

    unknown = sorted(str(key) for key in normalized if str(key) not in config)
    if unknown:
        raise ValueError(
            f"Unknown {name} config keys: {unknown}. "
            f"Allowed keys: {sorted(str(key) for key in config.keys())}"
        )

    config.update(normalized)
    return normalized


def apply_config_sources(
    defaults: Mapping[str, Any],
    *,
    name: str,
    default_file: Optional[str] = None,
    file_envs: Iterable[str] = (),
    json_envs: Iterable[str] = (),
    aliases: Optional[Mapping[str, str]] = None,
    print_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Load config defaults, then repo/env file configs, then env JSON overrides."""

    config: Dict[str, Any] = deepcopy(dict(defaults))
    applied: Dict[str, Dict[str, Any]] = {}

    def apply_payload(label: str, payload: Mapping[str, Any]) -> None:
        applied_payload = merge_known_keys(config, payload, name=name, aliases=aliases)
        if applied_payload:
            applied[label] = applied_payload

    if default_file:
        resolved_default = resolve_repo_path(default_file)
        if os.path.exists(resolved_default):
            merge_known_keys(config, load_json_object(resolved_default), name=name, aliases=aliases)

    for env_name in file_envs:
        path = str(os.environ.get(env_name, "")).strip()
        if path:
            apply_payload(env_name, load_json_object(path))

    for env_name in json_envs:
        raw = str(os.environ.get(env_name, "")).strip()
        if raw:
            apply_payload(env_name, parse_json_object(raw, env_name))

    if print_prefix and applied:
        print(
            f"{print_prefix} " + json.dumps(applied, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    return config
