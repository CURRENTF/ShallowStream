"""Reliability helpers shared by evaluation runners."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Any, Iterable, List, Optional


_OPTION_LABEL_RE = re.compile(r"^\s*([A-D])\.\s*(.*)$", re.DOTALL)
_UNSUCCESSFUL_STATUSES = {
    "invalid_input",
    "model_failed",
    "parse_failed",
    "preprocess_failed",
    "metric_failed",
    "skipped",
    "skipped_by_policy",
}
_EMPTY_MODEL_RESPONSE_ERROR = "model returned an empty response"


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    indent: int = 4,
    sort_keys: bool = False,
) -> None:
    """Atomically replace a JSON file without exposing a truncated destination."""

    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=indent,
                ensure_ascii=False,
                sort_keys=sort_keys,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_timestamp_seconds(value: Any) -> Optional[float]:
    """Parse seconds or ``HH:MM:SS`` timestamps; reject invalid/negative input."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(":")]
        if not 1 <= len(parts) <= 3:
            return None
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            return None
        if any(not math.isfinite(number) or number < 0 for number in numbers):
            return None
        if len(numbers) > 1 and any(number >= 60 for number in numbers[1:]):
            return None
        result = 0.0
        for index, number in enumerate(reversed(numbers)):
            result += number * (60 ** index)
    if not math.isfinite(result) or result < 0:
        return None
    return int(result) if result.is_integer() else result


def normalize_multiple_choice_options(options: Iterable[Any]) -> List[str]:
    """Return exactly one labelled A-D option, joining split annotation fragments."""

    raw_options = [str(value) for value in options]
    if len(raw_options) == 4:
        normalized = []
        for expected_label, raw in zip("ABCD", raw_options):
            text = raw.strip()
            match = _OPTION_LABEL_RE.match(text)
            content = match.group(2).strip() if match else text
            if not content:
                raise ValueError(f"empty option at position {expected_label}: {raw_options!r}")
            # Four-entry annotations are ordered choices. Canonicalize labels by
            # position so upstream typos such as A/B/B/D do not alter prompt
            # structure or silently drop a choice.
            normalized.append(f"{expected_label}. {content}")
        return normalized

    grouped = {}
    current_label = None
    for raw in raw_options:
        text = raw.strip()
        if not text:
            continue
        match = _OPTION_LABEL_RE.match(text)
        if match:
            label, content = match.groups()
            if label in grouped:
                raise ValueError(f"duplicate option label {label}: {raw_options!r}")
            grouped[label] = content.strip()
            current_label = label
            continue
        if current_label is None:
            raise ValueError(f"unlabelled option fragment before option A: {raw_options!r}")
        grouped[current_label] = f"{grouped[current_label]} {text}".strip()

    if set(grouped) != set("ABCD"):
        raise ValueError(f"expected exactly options A-D, found {sorted(grouped)}")
    return [f"{label}. {grouped[label]}" for label in "ABCD"]


def has_successful_answer(question: dict, model_name: str) -> bool:
    """Return whether a question contains a usable completed model response."""

    results = question.get("results")
    if isinstance(results, dict):
        status = str(results.get("status", "")).strip().lower()
        if status in _UNSUCCESSFUL_STATUSES:
            return False

    value = question.get(model_name)
    if isinstance(value, str):
        text = value.strip()
        return bool(text and text.upper() != "SKIPPED")
    if isinstance(value, dict):
        history = value.get("dialog_history")
        if isinstance(history, list):
            return any(
                isinstance(entry, dict)
                and entry.get("role") == "assistant"
                and str(entry.get("content", "")).strip()
                for entry in history
            )
        return bool(value)
    return False


def is_scorable_empty_model_response(question: dict, model_name: str) -> bool:
    """Return whether an explicit empty generation should count as incorrect."""

    if model_name not in question:
        return False
    results = question.get("results")
    if not isinstance(results, dict):
        return False
    if str(results.get("status", "")).strip().lower() != "model_failed":
        return False
    if str(results.get("error", "")).strip() != _EMPTY_MODEL_RESPONSE_ERROR:
        return False
    response = question.get(model_name)
    return response is None or (isinstance(response, str) and not response.strip())


def successful_result_metadata(response: Any, results: Any = None) -> dict:
    """Normalize model-returned metadata to an explicit success/failure status."""

    metadata = dict(results) if isinstance(results, dict) else {}
    if response is None or (isinstance(response, str) and not response.strip()):
        metadata["status"] = "model_failed"
        metadata.setdefault("error", _EMPTY_MODEL_RESPONSE_ERROR)
    else:
        metadata["status"] = "success"
    return metadata
