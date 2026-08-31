"""GPU and retained-KV measurements for controlled streaming profiles."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Sequence

import torch


def resolve_memory_profile_checkpoints(
    config: Mapping[str, Any],
    *,
    batch_frames: int,
) -> list[int]:
    raw = config.get("memory_profile_checkpoints_frames") or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("memory_profile_checkpoints_frames must be a list of frame counts")
    checkpoints = sorted({int(value) for value in raw})
    if any(value <= 0 for value in checkpoints):
        raise ValueError("memory_profile_checkpoints_frames must contain positive values")
    if checkpoints and int(batch_frames) <= 0:
        raise ValueError("memory profiling requires streaming_prefill_batch_frames > 0")
    misaligned = [value for value in checkpoints if value % int(batch_frames) != 0]
    if misaligned:
        raise ValueError(
            "memory profile checkpoints must align with streaming prefill batches: "
            f"batch_frames={batch_frames}, misaligned={misaligned}"
        )
    return checkpoints


def cache_kv_bytes(cache: Mapping[int, Mapping[str, Any]]) -> int:
    total = 0
    for entry in cache.values():
        for name in ("k", "v"):
            tensor = entry.get(name)
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel()) * int(tensor.element_size())
    return total


def cuda_allocator_snapshot() -> Dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA memory profiling requires an available CUDA device")
    torch.cuda.synchronize()
    return {
        "timestamp_ns": time.time_ns(),
        "process_pid": os.getpid(),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def begin_stream_memory_profile(
    config: Mapping[str, Any],
    *,
    model_load_baseline: Mapping[str, int],
) -> Dict[str, Any]:
    checkpoints = resolve_memory_profile_checkpoints(
        config,
        batch_frames=int(config.get("streaming_prefill_batch_frames", 0) or 0),
    )
    if not checkpoints:
        return {}
    empty_cache = bool(config.get("memory_profile_empty_cache_before_sample", False))
    if empty_cache:
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "checkpoints_frames": checkpoints,
        "empty_cache_before_sample": empty_cache,
        "model_load_baseline": dict(model_load_baseline),
        "sample_baseline": cuda_allocator_snapshot(),
        "checkpoints": [],
    }


def append_stream_memory_checkpoint(
    profile: Dict[str, Any],
    *,
    sampled_frames: int,
    session: Any,
) -> None:
    snapshot = cuda_allocator_snapshot()
    snapshot.update(
        {
            "sampled_frames": int(sampled_frames),
            "historical_kv_bytes": cache_kv_bytes(session.raw_lower_kv),
            "active_kv_bytes": cache_kv_bytes(session.active_lower_kv),
            "prefix_kv_bytes": cache_kv_bytes(session.prompt_prefix_lower_kv),
            "retained_temporal_units": len(session.frame_states),
            "retained_clusters": len(session.clusters),
        }
    )
    profile["checkpoints"].append(snapshot)


def validate_completed_memory_profile(profile: Mapping[str, Any]) -> None:
    expected: Sequence[int] = profile.get("checkpoints_frames") or []
    observed = [int(row["sampled_frames"]) for row in profile.get("checkpoints") or []]
    if observed != list(expected):
        raise RuntimeError(
            "Streaming memory profile did not reach every configured checkpoint: "
            f"expected={list(expected)}, observed={observed}"
        )
