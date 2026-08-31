"""Read-only decoded-frame cache contract for sharded StreamingBench lanes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from src.utils.routed_cache import acknowledge_cache_entry


_USE_COUNTS_LOCK = threading.Lock()
_USE_COUNTS_RAW = ""
_REMAINING_USES: Dict[str, int] = {}
DIRECT_SOURCE_RANGE_SAMPLING_CONTRACT = "global_frame_grid_pts_lte_v1"


def decode_cache_wait_enabled() -> bool:
    return (
        os.environ.get("STREAMINGBENCH_DECODE_CACHE_MISS_POLICY", "").strip().lower()
        == "wait"
    )


def decode_profile_from_env() -> Dict[str, Any]:
    raw = os.environ.get("STREAMINGBENCH_DECODE_PROFILE_JSON", "").strip()
    if not raw:
        raise RuntimeError(
            "STREAMINGBENCH_DECODE_PROFILE_JSON is required for decoded-cache consumers"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("STREAMINGBENCH_DECODE_PROFILE_JSON must be a JSON object")
    return normalize_decode_profile(payload)


def normalize_decode_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    sample_fps = float(profile.get("sample_fps", 0.0))
    if sample_fps <= 0:
        raise ValueError(f"StreamingBench sample_fps must be positive, got {sample_fps}")
    raw_max = profile.get("max_frames_num")
    max_frames = None if raw_max is None else int(raw_max)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("StreamingBench max_frames_num must be null or positive")
    artifact_format = str(profile.get("artifact_format", "raw_rgb") or "raw_rgb").strip()
    if artifact_format not in {"raw_rgb", "qwen_fetch_video"}:
        raise ValueError(
            "StreamingBench artifact_format must be raw_rgb or qwen_fetch_video"
        )
    normalized: Dict[str, Any] = {
        "sample_fps": sample_fps,
        "max_frames_num": max_frames,
        "artifact_format": artifact_format,
    }
    raw_range_backend = profile.get("range_decode_backend")
    if raw_range_backend is not None:
        range_backend = str(raw_range_backend).strip().lower()
        if range_backend not in {"transcoded_clip", "direct_source_range"}:
            raise ValueError(
                "StreamingBench range_decode_backend must be transcoded_clip "
                "or direct_source_range"
            )
        normalized["range_decode_backend"] = range_backend
        if range_backend == "direct_source_range":
            normalized["sampling_contract"] = DIRECT_SOURCE_RANGE_SAMPLING_CONTRACT
    if artifact_format == "qwen_fetch_video":
        normalized["image_patch_size"] = int(profile.get("image_patch_size", 16) or 16)
        normalized["frame_max_pixels"] = int(profile.get("frame_max_pixels", 0) or 0)
        normalized["video_total_pixels"] = int(profile.get("video_total_pixels", 0) or 0)
        if normalized["image_patch_size"] <= 0:
            raise ValueError("Qwen artifact image_patch_size must be positive")
        if normalized["frame_max_pixels"] < 0 or normalized["video_total_pixels"] < 0:
            raise ValueError("Qwen artifact pixel budgets must not be negative")

    # SimpleStream adapters only feed a bounded recent-frame window to the
    # model.  Keep that producer-side contract in the decode profile so the
    # shared producer can avoid materialising the older sampled frames.  The
    # model-specific config names are accepted here because OneVision passes
    # its resolved config directly while Qwen uses fetch_video_recent_frames.
    raw_recent = None
    for key in ("recent_frames", "fetch_video_recent_frames", "input_recent_frames"):
        if key in profile and profile[key] is not None:
            raw_recent = profile[key]
            break
    recent_frames = int(raw_recent or 0)
    if recent_frames < 0:
        raise ValueError("StreamingBench recent frame count must be non-negative")
    if recent_frames > 0:
        raw_duration = profile.get(
            "recent_chunk_duration",
            profile.get("chunk_duration", 1.0),
        )
        recent_chunk_duration = float(raw_duration or 1.0)
        if recent_chunk_duration <= 0:
            raise ValueError("StreamingBench recent chunk duration must be positive")
        normalized["recent_frames"] = recent_frames
        normalized["recent_chunk_duration"] = recent_chunk_duration
    return normalized


def _canonical_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"clip timestamp must be non-negative, got {value!r}")
    return format(number, ".9g")


def decode_cache_key(
    video_file: str,
    start_time: Any,
    end_time: Any,
    profile: Mapping[str, Any],
    *,
    namespace: str,
) -> str:
    start = float(start_time)
    end = float(end_time)
    if end <= start:
        raise ValueError(f"decoded-cache range must be increasing: start={start}, end={end}")
    source = os.path.realpath(os.path.abspath(video_file))
    try:
        stat = os.stat(source)
        source_stat: Optional[Dict[str, int]] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        source_stat = None
    payload = {
        "schema_version": 1,
        "namespace": str(namespace),
        "source_path": source,
        "source_stat": source_stat,
        "start_time": _canonical_number(start),
        "end_time": _canonical_number(end),
        "profile": normalize_decode_profile(profile),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_cache_paths(cache_dir: str, cache_key: str) -> Tuple[str, str]:
    data_path = os.path.join(os.path.abspath(cache_dir), f"{cache_key}.npz")
    return data_path, f"{data_path}.json"


def archive_stale_producer_manifest(
    producer_manifest: str,
    *,
    run_root: str,
    route_run_id: str,
) -> Optional[str]:
    """Atomically preserve a previous producer status before a resumed attempt."""

    source = os.path.abspath(producer_manifest)
    if not os.path.isfile(source):
        return None
    history_dir = os.path.join(os.path.abspath(run_root), "producer_manifest_history")
    os.makedirs(history_dir, exist_ok=True)
    archive = os.path.join(history_dir, f"{route_run_id}.json")
    if os.path.exists(archive):
        raise FileExistsError(f"Producer manifest archive already exists: {archive}")
    try:
        os.replace(source, archive)
    except FileNotFoundError:
        return None
    return archive


def decoded_cache_location(
    video_file: str,
    start_time: Any,
    end_time: Any,
    *,
    cache_dir: Optional[str] = None,
    profile: Optional[Mapping[str, Any]] = None,
    namespace: Optional[str] = None,
) -> Tuple[str, str, str]:
    resolved_cache_dir = (
        str(cache_dir).strip()
        if cache_dir is not None
        else os.environ.get("STREAMINGBENCH_DECODE_CACHE_DIR", "").strip()
    )
    if not resolved_cache_dir:
        raise RuntimeError(
            "STREAMINGBENCH_DECODE_CACHE_DIR is required for decoded-cache consumers"
        )
    resolved_profile = (
        normalize_decode_profile(profile) if profile is not None else decode_profile_from_env()
    )
    resolved_namespace = (
        str(namespace).strip()
        if namespace is not None
        else os.environ.get("STREAMINGBENCH_DECODE_CACHE_NAMESPACE", "").strip()
    )
    if not resolved_namespace:
        raise RuntimeError(
            "STREAMINGBENCH_DECODE_CACHE_NAMESPACE is required for decoded-cache consumers"
        )
    key = decode_cache_key(
        video_file,
        start_time,
        end_time,
        resolved_profile,
        namespace=resolved_namespace,
    )
    data_path, metadata_path = decode_cache_paths(resolved_cache_dir, key)
    return key, data_path, metadata_path


def _load_ready_metadata(
    data_path: str,
    metadata_path: str,
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(data_path) or os.path.getsize(data_path) <= 0:
        return None
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict) or metadata.get("cache_key") != cache_key:
        return None
    return metadata if metadata.get("status") == "ready" else None


def _ready_cache_result(data_path: str, metadata: Mapping[str, Any]) -> str:
    if metadata.get("artifact_kind") == "no_new_video_chunk":
        acknowledge_decoded_cache(data_path)
        return "__NO_NEW_VIDEO_CHUNK__"
    return data_path


def find_ready_decoded_cache(video_file: str, start_time: Any, end_time: Any) -> Optional[str]:
    if not decode_cache_wait_enabled():
        return None
    key, data_path, metadata_path = decoded_cache_location(video_file, start_time, end_time)
    metadata = _load_ready_metadata(data_path, metadata_path, key)
    return None if metadata is None else _ready_cache_result(data_path, metadata)


def wait_for_decoded_cache(video_file: str, start_time: Any, end_time: Any) -> str:
    key, data_path, metadata_path = decoded_cache_location(video_file, start_time, end_time)
    timeout = float(os.environ.get("STREAMINGBENCH_DECODE_CACHE_WAIT_TIMEOUT_SECONDS", "1800"))
    poll = float(os.environ.get("STREAMINGBENCH_DECODE_CACHE_POLL_SECONDS", "0.1"))
    if timeout <= 0 or poll <= 0:
        raise ValueError("decoded-cache wait timeout and poll interval must be positive")
    producer_manifest = os.environ.get(
        "STREAMINGBENCH_DECODE_CACHE_PRODUCER_MANIFEST", ""
    ).strip()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metadata = _load_ready_metadata(data_path, metadata_path, key)
        if metadata is not None:
            return _ready_cache_result(data_path, metadata)
        if producer_manifest and os.path.isfile(producer_manifest):
            try:
                with open(producer_manifest, "r", encoding="utf-8") as handle:
                    producer = json.load(handle)
                if producer.get("status") == "failed":
                    raise RuntimeError(
                        "StreamingBench decode producer failed while waiting for "
                        f"cache_key={key}: {producer.get('error', 'unknown failure')}"
                    )
            except (FileNotFoundError, json.JSONDecodeError):
                # The manifest is atomically replaced on a shared filesystem.
                # A reader may briefly observe ENOENT between directory-entry
                # refreshes; the next poll will see either the new manifest or
                # the requested artifact.
                pass
        time.sleep(poll)
    raise TimeoutError(
        "Timed out waiting for StreamingBench decoded cache: "
        f"cache_key={key}, path={data_path}, timeout_seconds={timeout}"
    )


def load_decoded_cache_metadata(data_path: str) -> Dict[str, Any]:
    metadata_path = f"{os.path.abspath(data_path)}.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise ValueError(f"Invalid StreamingBench decoded-cache metadata: {metadata_path}")
    return payload


def decoded_cache_audio_source(data_path: str) -> Optional[str]:
    if not str(data_path).lower().endswith(".npz"):
        return None
    try:
        metadata = load_decoded_cache_metadata(data_path)
    except (OSError, ValueError):
        return None
    clip_path = str(metadata.get("source_clip_path", "")).strip()
    return clip_path if clip_path and os.path.isfile(clip_path) else None


def _refresh_use_counts() -> None:
    global _USE_COUNTS_RAW, _REMAINING_USES
    raw = os.environ.get("STREAMINGBENCH_DECODE_CACHE_USE_COUNTS_JSON", "").strip()
    if raw == _USE_COUNTS_RAW:
        return
    payload = json.loads(raw) if raw else {}
    if not isinstance(payload, dict):
        raise ValueError("STREAMINGBENCH_DECODE_CACHE_USE_COUNTS_JSON must be an object")
    counts = {str(key): int(value) for key, value in payload.items()}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("StreamingBench decoded-cache use counts must be positive")
    _USE_COUNTS_RAW = raw
    _REMAINING_USES = counts


def acknowledge_decoded_cache(data_path: str) -> Optional[str]:
    """Acknowledge after the final planned read of a decoded cache entry."""
    if not str(data_path).lower().endswith(".npz"):
        return None
    state_dir = os.environ.get("STREAMINGBENCH_DECODE_CACHE_ROUTE_STATE_DIR", "").strip()
    consumer_id = os.environ.get("STREAMINGBENCH_DECODE_CACHE_CONSUMER_ID", "").strip()
    if not state_dir and not consumer_id:
        return None
    if not state_dir or not consumer_id:
        raise RuntimeError(
            "STREAMINGBENCH_DECODE_CACHE_ROUTE_STATE_DIR and "
            "STREAMINGBENCH_DECODE_CACHE_CONSUMER_ID must be configured together"
        )
    try:
        metadata = load_decoded_cache_metadata(data_path)
    except FileNotFoundError:
        # A producer may evict an entry after observing the first (valid)
        # acknowledgement while a same-timestamp replay sends the planned
        # acknowledgement a second time.  The cache key is the artifact
        # basename, so this late idempotent acknowledgement does not need the
        # already-evicted sidecar metadata.
        cache_key = os.path.splitext(os.path.basename(os.path.abspath(data_path)))[0]
        if not cache_key:
            raise
    else:
        cache_key = str(metadata.get("cache_key", "")).strip()
        if not cache_key:
            raise ValueError(f"Decoded-cache metadata has no cache_key: {data_path}.json")
    with _USE_COUNTS_LOCK:
        _refresh_use_counts()
        remaining = _REMAINING_USES.get(cache_key, 1)
        if remaining > 1:
            _REMAINING_USES[cache_key] = remaining - 1
            return None
        _REMAINING_USES.pop(cache_key, None)
    acknowledge_cache_entry(state_dir, cache_key, consumer_id)
    return cache_key


def acknowledge_planned_cache_key(cache_key: str) -> None:
    state_dir = os.environ.get("STREAMINGBENCH_DECODE_CACHE_ROUTE_STATE_DIR", "").strip()
    consumer_id = os.environ.get("STREAMINGBENCH_DECODE_CACHE_CONSUMER_ID", "").strip()
    if state_dir and consumer_id:
        acknowledge_cache_entry(state_dir, str(cache_key), consumer_id)


def clear_consumer_use_count_state() -> None:
    """Test helper and worker reset hook."""
    global _USE_COUNTS_RAW, _REMAINING_USES
    with _USE_COUNTS_LOCK:
        _USE_COUNTS_RAW = ""
        _REMAINING_USES = {}
