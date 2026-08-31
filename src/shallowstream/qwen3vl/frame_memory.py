"""Video sampling and frame-level memory for ShallowStream Qwen3-VL V3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import fetch_video

from src.utils.routed_cache import acknowledge_cache_entry
from src.utils.streamingbench_decode_cache import load_decoded_cache_metadata

from .config import MODEL_NAME, _as_bool, _as_float, _as_int


_CONFIGURED_DECORD_THREADS: Optional[int] = None


def _uniform_frame_indices(start_frame: int, end_frame: int, nframes: int) -> List[int]:
    if nframes <= 0:
        raise ValueError("uniform frame count must be positive")
    if end_frame < start_frame:
        raise ValueError("uniform frame range must be non-empty")
    return torch.linspace(start_frame, end_frame, nframes).round().long().tolist()


def _fetch_video_decord_chunked(
    video_request: Dict[str, Any],
    image_patch_size: int,
    *,
    decode_threads: int,
    chunk_frames: int,
    recent_frames: int = 0,
    recent_chunk_duration: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Match Qwen's Decord path without retaining every source-sized frame.

    ``recent_frames`` is an explicit SimpleStream optimization: calculate the
    same uniform source-frame indices and spatial budget as the full Qwen path,
    then decode only the final ``recent_frames`` temporal chunks that the
    baseline actually feeds to the model.  A chunk can contain multiple
    sampled frames when the source sampling rate is higher than one FPS.
    A value of zero preserves the historical full-video behavior.
    """

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if recent_frames < 0:
        raise ValueError("recent_frames must be non-negative")
    if recent_chunk_duration <= 0:
        raise ValueError("recent_chunk_duration must be positive")

    import decord
    from qwen_vl_utils import vision_process

    reader_kwargs = {"num_threads": int(decode_threads)} if decode_threads > 0 else {}
    reader = decord.VideoReader(str(video_request["video"]), **reader_kwargs)
    source_frame_count = len(reader)
    video_fps = float(reader.get_avg_fps())
    start_frame, end_frame, ranged_frame_count = (
        vision_process.calculate_video_frame_range(
            video_request,
            source_frame_count,
            video_fps,
        )
    )
    nframes = int(
        vision_process.smart_nframes(
            video_request,
            total_frames=ranged_frame_count,
            video_fps=video_fps,
        )
    )
    full_indices = _uniform_frame_indices(start_frame, end_frame, nframes)
    if len(full_indices) > 1:
        full_frame_dt = max(
            float(full_indices[-1] - full_indices[-2]) / max(video_fps, 1e-6),
            1.0 / max(video_fps, 1e-6),
        )
    else:
        full_frame_dt = 1.0 / max(video_fps, 1e-6)
    full_last_timestamp = float(full_indices[-1]) / max(video_fps, 1e-6)
    full_video_duration_s = min(
        (math.floor(full_last_timestamp / recent_chunk_duration) + 1.0)
        * recent_chunk_duration,
        full_last_timestamp + full_frame_dt,
    )
    indices = full_indices
    if recent_frames > 0:
        chunk_ids = [
            int(math.floor((float(index) / max(video_fps, 1e-6)) / recent_chunk_duration))
            for index in indices
        ]
        retained_chunks = set(sorted(set(chunk_ids))[-int(recent_frames) :])
        indices = [
            index
            for index, chunk_id in zip(indices, chunk_ids)
            if chunk_id in retained_chunks
        ]

    image_factor = int(image_patch_size) * int(vision_process.SPATIAL_MERGE_SIZE)
    min_pixels = video_request.get(
        "min_pixels",
        vision_process.VIDEO_MIN_TOKEN_NUM * image_factor * image_factor,
    )
    total_pixels = video_request.get(
        "total_pixels",
        vision_process.MODEL_SEQ_LEN * image_factor * image_factor * 0.9,
    )
    max_pixels = max(
        min(
            vision_process.VIDEO_MAX_TOKEN_NUM * image_factor * image_factor,
            total_pixels / nframes * vision_process.FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    max_pixels = min(video_request.get("max_pixels", max_pixels), max_pixels)

    resized_chunks: List[torch.Tensor] = []
    output_size: Optional[Tuple[int, int]] = None
    for offset in range(0, len(indices), chunk_frames):
        chunk_indices = indices[offset : offset + chunk_frames]
        source_frames = reader.get_batch(chunk_indices).asnumpy()
        chunk = torch.tensor(source_frames).permute(0, 3, 1, 2)
        if output_size is None:
            source_height, source_width = int(chunk.shape[-2]), int(chunk.shape[-1])
            if "resized_height" in video_request and "resized_width" in video_request:
                output_size = tuple(
                    int(value)
                    for value in vision_process.smart_resize(
                        video_request["resized_height"],
                        video_request["resized_width"],
                        factor=image_factor,
                    )
                )
            else:
                output_size = tuple(
                    int(value)
                    for value in vision_process.smart_resize(
                        source_height,
                        source_width,
                        factor=image_factor,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                )
        resized_chunks.append(
            vision_process.transforms.functional.resize(
                chunk,
                list(output_size),
                interpolation=vision_process.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        del chunk, source_frames

    video = torch.cat(resized_chunks, dim=0).float()
    metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": ranged_frame_count,
        "video_backend": "decord",
    }
    if recent_frames > 0:
        metadata.update(
            {
                "full_sampled_frames": nframes,
                "full_video_duration_s": full_video_duration_s,
                "recent_frames_limit": int(recent_frames),
                "recent_chunk_duration": float(recent_chunk_duration),
            }
        )
    return video, metadata


def _configure_decord_threads(
    num_threads: int,
    *,
    exact_uniform_sampling: bool = False,
) -> None:
    """Install the audited decord reader for threads or exact uniform sampling."""
    global _CONFIGURED_DECORD_THREADS
    if num_threads <= 0 and not exact_uniform_sampling:
        return
    effective_threads = max(int(num_threads), 0)
    if _CONFIGURED_DECORD_THREADS == effective_threads:
        return
    if _CONFIGURED_DECORD_THREADS is not None:
        raise RuntimeError(
            "A process cannot mix Qwen decord thread counts: "
            f"configured={_CONFIGURED_DECORD_THREADS}, requested={effective_threads}"
        )

    import decord
    from qwen_vl_utils import vision_process

    if exact_uniform_sampling and vision_process.get_video_reader_backend() != "decord":
        raise RuntimeError(
            "exact_uniform_sampled_frames requires the qwen-vl-utils decord backend"
        )

    def _read_video_decord_limited(ele: Dict[str, Any]):
        video_path = ele["video"]
        started = time.time()
        reader_kwargs = (
            {"num_threads": effective_threads}
            if effective_threads > 0
            else {}
        )
        reader = decord.VideoReader(video_path, **reader_kwargs)
        total_frames, video_fps = len(reader), reader.get_avg_fps()
        start_frame, end_frame, total_frames = vision_process.calculate_video_frame_range(
            ele,
            total_frames,
            video_fps,
        )
        if "nframes" in ele:
            nframes = int(ele["nframes"])
            frame_factor = int(vision_process.FRAME_FACTOR)
            if nframes < frame_factor or nframes % frame_factor != 0:
                raise ValueError(
                    "exact uniform nframes must be a positive multiple of "
                    f"Qwen FRAME_FACTOR={frame_factor}; got {nframes}"
                )
            sampling_mode = "exact_uniform"
        else:
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=total_frames,
                video_fps=video_fps,
            )
            sampling_mode = "fps"
        indices = _uniform_frame_indices(start_frame, end_frame, nframes)
        video = torch.tensor(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        vision_process.logger.info(
            "decord(num_threads=%s,sampling=%s): video_path=%s, total_frames=%s, "
            "video_fps=%s, sampled_frames=%s, unique_indices=%s, time=%.3fs",
            effective_threads or "auto",
            sampling_mode,
            video_path,
            total_frames,
            video_fps,
            nframes,
            len(set(indices)),
            time.time() - started,
        )
        metadata = {
            "fps": video_fps,
            "frames_indices": indices,
            "total_num_frames": total_frames,
            "video_backend": "decord",
        }
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        return video, metadata, sample_fps

    vision_process.VIDEO_READER_BACKENDS["decord"] = _read_video_decord_limited
    _CONFIGURED_DECORD_THREADS = effective_threads


def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _pil_to_embedding(frame: Image.Image, size: int) -> np.ndarray:
    image = frame.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Light-weight visual descriptor: color layout + first-order gradients.
    gx = np.diff(arr, axis=1, append=arr[:, -1:, :])
    gy = np.diff(arr, axis=0, append=arr[-1:, :, :])
    feat = np.concatenate([arr.reshape(-1), gx.reshape(-1), gy.reshape(-1)])
    return _l2_normalize(feat)

def _resize_for_center(frame: Image.Image, size: int) -> np.ndarray:
    image = frame.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32)

def _resize_memory_image(frame: Image.Image, size: int) -> Image.Image:
    return frame.convert("RGB").resize((size, size), Image.BICUBIC)

@dataclass
class SampledFrame:
    index: int
    timestamp: float
    image: Image.Image
    embedding: np.ndarray
    # Qwen's temporal patching can merge multiple sampled frames into one
    # retrieval unit. Keep the original frames so a selected recent unit can
    # still reconstruct the intended raw-frame prompt.
    source_frames: Optional[List["SampledFrame"]] = None

@dataclass
class SampledVideo:
    frames: List[SampledFrame]
    video: torch.Tensor
    metadata: Dict[str, Any]

@dataclass
class ClusterCenter:
    start_index: int
    end_index: int
    count: int
    centroid: np.ndarray
    center_image: np.ndarray
    start_time: float
    end_time: float

    @classmethod
    def from_frame(cls, frame: SampledFrame, center_image_size: int) -> "ClusterCenter":
        return cls(
            start_index=frame.index,
            end_index=frame.index,
            count=1,
            centroid=frame.embedding.copy(),
            center_image=_resize_for_center(frame.image, center_image_size),
            start_time=frame.timestamp,
            end_time=frame.timestamp,
        )

    def merge(self, frame: SampledFrame, center_image_size: int) -> None:
        new_count = self.count + 1
        self.centroid = _l2_normalize((self.centroid * self.count + frame.embedding) / new_count)
        self.center_image = (self.center_image * self.count + _resize_for_center(frame.image, center_image_size)) / new_count
        self.count = new_count
        self.end_index = frame.index
        self.end_time = frame.timestamp

    def to_image(self) -> Image.Image:
        arr = np.clip(self.center_image, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

class Qwen3VLFrameMemory:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        exact_uniform_frames = int(config.get("exact_uniform_sampled_frames", 0) or 0)
        max_sampled_frames = int(config.get("max_sampled_frames", 0) or 0)
        if exact_uniform_frames > 0 and max_sampled_frames > 0:
            raise ValueError(
                "exact_uniform_sampled_frames and max_sampled_frames are mutually exclusive"
            )
        temporal_factor = max(int(config.get("video_temporal_patch_size", 2) or 2), 1)
        if exact_uniform_frames > 0 and exact_uniform_frames % temporal_factor != 0:
            raise ValueError(
                "exact_uniform_sampled_frames must be divisible by "
                f"video_temporal_patch_size={temporal_factor}"
            )
        _configure_decord_threads(
            int(config.get("video_decode_threads", 0) or 0),
            exact_uniform_sampling=exact_uniform_frames > 0,
        )
        self.embedding_size = _as_int(config, "frame_embedding_size")
        self.center_image_size = _as_int(config, "cluster_center_image_size")
        self.recent_frames = _as_int(config, "retrieval_recent_units")
        self.long_topk = _as_int(config, "long_cluster_topk")
        self.cluster_threshold = _as_float(config, "cluster_threshold")
        self.cache_write_failures: List[Dict[str, str]] = []
        raw_use_counts = config.get("fetch_video_cache_consumer_use_counts", {})
        if not isinstance(raw_use_counts, dict):
            raise ValueError("fetch_video_cache_consumer_use_counts must be an object")
        self._fetch_video_cache_remaining_loads = {
            os.path.abspath(str(path)): int(count)
            for path, count in raw_use_counts.items()
        }
        if any(count <= 0 for count in self._fetch_video_cache_remaining_loads.values()):
            raise ValueError("fetch_video_cache_consumer_use_counts values must be positive")
        self._fetch_video_reuse_cache: Dict[str, SampledVideo] = {}
        self._fetch_video_reuse_lock = threading.Lock()
        self._fetch_video_loaded_paths = set()
        self._fetch_video_acknowledged_paths = set()

    def _video_request(self, video_path: str, video_start: Optional[float] = None) -> Dict[str, Any]:
        request: Dict[str, Any] = {"video": video_path}
        total_pixels = int(self.config.get("video_total_pixels", 0) or 0)
        if total_pixels > 0:
            request["total_pixels"] = total_pixels
        frame_max_pixels = int(self.config.get("frame_max_pixels", 0) or 0)
        if frame_max_pixels > 0:
            request["max_pixels"] = frame_max_pixels

        exact_uniform_frames = int(
            self.config.get("exact_uniform_sampled_frames", 0) or 0
        )
        if exact_uniform_frames > 0:
            request["nframes"] = int(exact_uniform_frames)
        else:
            request["fps"] = max(_as_float(self.config, "sample_fps"), 1e-6)
            max_frames = _as_int(self.config, "max_sampled_frames")
            # qwen-vl-utils has its own default max-frame cap. For ShallowStream
            # we want every fps-sampled frame unless the user explicitly caps it.
            request["max_frames"] = (
                int(max_frames) if max_frames > 0 else 1_000_000_000
            )
        if video_start is not None:
            request["video_start"] = max(0.0, float(video_start))
        return request

    def _frame_indices_from_metadata(self, metadata: Dict[str, Any], frame_count: int) -> Tuple[List[int], float]:
        raw_fps = max(float(metadata.get("fps", self.config.get("sample_fps", 1.0)) or 1.0), 1e-6)
        frame_indices = metadata.get("frames_indices")
        if isinstance(frame_indices, torch.Tensor):
            frame_indices = frame_indices.detach().cpu().reshape(-1).tolist()
        elif frame_indices is not None and not isinstance(frame_indices, (list, tuple)):
            try:
                frame_indices = list(frame_indices)
            except TypeError:
                frame_indices = None
        if frame_indices is None or len(frame_indices) != frame_count:
            frame_indices = list(range(frame_count))
        return [int(idx) for idx in frame_indices], raw_fps

    def _fetch_video_cache_dir(self) -> Optional[str]:
        cache_dir = str(self.config.get("fetch_video_cache_dir", "") or "").strip()
        return cache_dir or None

    def _fetch_video_cache_storage_dtype(self) -> str:
        value = str(self.config.get("fetch_video_cache_storage_dtype", "original") or "original").strip().lower()
        if value not in {"original", "uint8"}:
            raise ValueError("fetch_video_cache_storage_dtype must be 'original' or 'uint8'")
        return value

    def _fetch_video_cache_key(self, video_request: Dict[str, Any], image_patch_size: int) -> str:
        video_path = str(video_request.get("video", ""))
        try:
            stat = os.stat(video_path)
            video_stat: Dict[str, object] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except OSError:
            video_stat = {}
        try:
            qwen_vl_utils_version = importlib_metadata.version("qwen-vl-utils")
        except importlib_metadata.PackageNotFoundError:
            qwen_vl_utils_version = "unknown"
        payload = {
            "fetch_video_request": video_request,
            "image_patch_size": int(image_patch_size),
            "return_video_metadata": True,
            "qwen_vl_utils_version": qwen_vl_utils_version,
            "video_stat": video_stat,
            "storage_dtype": self._fetch_video_cache_storage_dtype(),
        }
        recent_frames = int(self.config.get("fetch_video_recent_frames", 0) or 0)
        if recent_frames > 0:
            payload.update(
                {
                    "recent_frames_limit": recent_frames,
                    "recent_chunk_duration": float(
                        self.config.get("chunk_duration", 1.0) or 1.0
                    ),
                }
            )
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _load_fetch_video_cache(self, cache_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Safe cache loading requires a PyTorch version with weights_only support"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load fetch_video cache: {cache_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid fetch_video cache payload: {cache_path}")
        video = payload.get("video")
        metadata = payload.get("metadata")
        if not isinstance(video, torch.Tensor) or not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid fetch_video cache tensors/metadata: {cache_path}")
        return video, dict(metadata)

    def _save_fetch_video_cache(
        self,
        cache_path: str,
        video: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            storage_dtype = self._fetch_video_cache_storage_dtype()
            stored_video = video
            if storage_dtype == "uint8" and video.dtype != torch.uint8:
                compact = video.to(torch.uint8)
                if not torch.equal(video, compact.to(dtype=video.dtype)):
                    raise ValueError(
                        "fetch_video tensor cannot round-trip exactly through uint8 cache storage"
                    )
                stored_video = compact
            payload = {
                "video": stored_video,
                "metadata": dict(metadata),
                "storage_dtype": storage_dtype,
                "source_dtype": str(video.dtype),
            }
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            failure = {"cache_path": cache_path, "error": f"{type(exc).__name__}: {exc}"}
            self.cache_write_failures.append(failure)
            if _as_bool(self.config, "fetch_video_cache_write_strict"):
                raise RuntimeError(f"Failed to save fetch_video cache: {cache_path}") from exc
            print(
                f"[{MODEL_NAME}] cache write failed; continuing with decoded tensor: {failure}",
                flush=True,
            )

    @staticmethod
    def _fetch_video_uncached(
        video_request: Dict[str, Any],
        image_patch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        video, metadata = fetch_video(
            video_request,
            image_patch_size=int(image_patch_size),
            return_video_metadata=True,
        )
        return video, dict(metadata if isinstance(metadata, dict) else {})

    def _wait_for_fetch_video_cache(
        self,
        cache_path: str,
        *,
        video_path: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        timeout = float(self.config.get("fetch_video_cache_wait_timeout_seconds", 1800.0) or 0.0)
        poll = float(self.config.get("fetch_video_cache_wait_poll_seconds", 0.1) or 0.0)
        if timeout <= 0:
            raise ValueError("fetch_video_cache_wait_timeout_seconds must be positive")
        if poll <= 0:
            raise ValueError("fetch_video_cache_wait_poll_seconds must be positive")

        deadline = time.monotonic() + timeout
        while True:
            if os.path.isfile(cache_path):
                return self._load_fetch_video_cache(cache_path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for the decoded-video producer: "
                    f"video={video_path}, cache={cache_path}, timeout_seconds={timeout}"
                )
            time.sleep(min(poll, remaining))

    def _fetch_video_cache_location(self, video_path: str) -> Tuple[str, str]:
        cache_dir = self._fetch_video_cache_dir()
        if not cache_dir:
            raise ValueError("fetch_video cache location requires fetch_video_cache_dir")
        video_request = self._video_request(video_path)
        image_patch_size = int(self.config.get("video_image_patch_size", 16) or 16)
        cache_key = self._fetch_video_cache_key(video_request, image_patch_size)
        return cache_key, os.path.join(cache_dir, f"{cache_key}.pt")

    def acknowledge_cached_video(self, video_path: str) -> Optional[str]:
        state_dir = str(
            self.config.get("fetch_video_cache_route_state_dir", "") or ""
        ).strip()
        consumer_id = str(
            self.config.get("fetch_video_cache_consumer_id", "") or ""
        ).strip()
        if not state_dir and not consumer_id:
            return None
        if not state_dir or not consumer_id:
            raise ValueError(
                "fetch_video_cache_route_state_dir and "
                "fetch_video_cache_consumer_id must be configured together"
            )
        absolute_path = os.path.abspath(video_path)
        if absolute_path not in self._fetch_video_cache_remaining_loads:
            return None
        if absolute_path in self._fetch_video_acknowledged_paths:
            return None
        if (
            self._fetch_video_cache_remaining_loads[absolute_path] > 1
            and absolute_path not in self._fetch_video_loaded_paths
        ):
            # A resumed journal entry is acknowledged without running model
            # inference. Retain one decoded copy first when another local
            # case will reuse the same entry after the producer evicts it.
            self.load_sampled_video(video_path)
        self._fetch_video_acknowledged_paths.add(absolute_path)
        cache_key, _cache_path = self._fetch_video_cache_location(video_path)
        acknowledge_cache_entry(state_dir, cache_key, consumer_id)
        return cache_key

    def prefetch_video(
        self,
        video_path: str,
        *,
        validate_cache_hit: bool = True,
    ) -> Dict[str, Any]:
        """Decode one video into the shared fetch-video cache without model work."""

        cache_dir = self._fetch_video_cache_dir()
        if not cache_dir:
            raise ValueError("prefetch_video requires fetch_video_cache_dir")
        if _as_bool(self.config, "fetch_video_cache_read_only"):
            raise ValueError("prefetch_video cannot use a read-only fetch-video cache")

        os.makedirs(cache_dir, exist_ok=True)
        video_request = self._video_request(video_path)
        image_patch_size = int(self.config.get("video_image_patch_size", 16) or 16)
        cache_key = self._fetch_video_cache_key(video_request, image_patch_size)
        cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
        if os.path.isfile(cache_path):
            sampled_frames = None
            if validate_cache_hit:
                video, _ = self._load_fetch_video_cache(cache_path)
                sampled_frames = int(video.shape[0])
            return {
                "cache_key": cache_key,
                "cache_path": cache_path,
                "cache_hit": True,
                "sampled_frames": sampled_frames,
            }

        decode_chunk_frames = int(
            self.config.get("fetch_video_prefetch_decode_chunk_frames", 0) or 0
        )
        recent_frames = int(self.config.get("fetch_video_recent_frames", 0) or 0)
        if recent_frames < 0:
            raise ValueError("fetch_video_recent_frames must be non-negative")
        if recent_frames > 0 and decode_chunk_frames <= 0:
            decode_chunk_frames = 16
        if decode_chunk_frames > 0:
            video, metadata = _fetch_video_decord_chunked(
                video_request,
                image_patch_size,
                decode_threads=int(self.config.get("video_decode_threads", 0) or 0),
                chunk_frames=int(decode_chunk_frames),
                recent_frames=recent_frames,
                recent_chunk_duration=float(
                    self.config.get("chunk_duration", 1.0) or 1.0
                ),
            )
        else:
            video, metadata = self._fetch_video_uncached(video_request, image_patch_size)
        self._save_fetch_video_cache(cache_path, video, metadata)
        if not os.path.isfile(cache_path):
            raise RuntimeError(f"Decoded-video cache entry was not published: {cache_path}")
        return {
            "cache_key": cache_key,
            "cache_path": cache_path,
            "cache_hit": False,
            "sampled_frames": int(video.shape[0]),
        }

    def _fetch_video_exact(self, video_request: Dict[str, Any], image_patch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cache_dir = self._fetch_video_cache_dir()
        if not cache_dir:
            return self._fetch_video_uncached(video_request, image_patch_size)

        cache_read_only = _as_bool(self.config, "fetch_video_cache_read_only")
        if not cache_read_only:
            os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._fetch_video_cache_key(video_request, int(image_patch_size))
        cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
        if os.path.exists(cache_path):
            return self._load_fetch_video_cache(cache_path)

        miss_policy = str(self.config.get("fetch_video_cache_miss_policy", "decode") or "decode").strip().lower()
        if miss_policy == "wait":
            return self._wait_for_fetch_video_cache(
                cache_path,
                video_path=str(video_request.get("video", "")),
            )
        if miss_policy == "error":
            raise FileNotFoundError(
                "Decoded-video cache miss under error policy: "
                f"video={video_request.get('video', '')}, cache={cache_path}"
            )
        if miss_policy != "decode":
            raise ValueError(
                "fetch_video_cache_miss_policy must be one of decode, wait, error; "
                f"got {miss_policy!r}"
            )

        video, metadata = self._fetch_video_uncached(video_request, image_patch_size)
        if not cache_read_only:
            self._save_fetch_video_cache(cache_path, video, metadata)
        return video, metadata

    def _decode_video_with_fetch_video(self, video_path: str, video_start: Optional[float] = None) -> SampledVideo:
        video_request = self._video_request(video_path, video_start=video_start)
        image_patch_size = int(self.config.get("video_image_patch_size", 16) or 16)
        video, metadata = self._fetch_video_exact(video_request, image_patch_size)
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError(f"Unexpected Qwen fetch_video output for video: {video_path}")

        metadata = dict(metadata)
        frame_count = int(video.shape[0])
        frame_indices, raw_fps = self._frame_indices_from_metadata(metadata, frame_count)
        compute_embeddings = self.config.get("compute_frame_embeddings", True)
        if isinstance(compute_embeddings, str):
            compute_embeddings = compute_embeddings.lower() in {"1", "true", "yes", "y", "on"}

        if video.dtype == torch.uint8:
            video_array = video.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        else:
            video_array = video.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        frames: List[SampledFrame] = []
        for frame_index, arr in zip(frame_indices, video_array):
            image = Image.fromarray(arr).convert("RGB")
            timestamp = float(frame_index) / raw_fps
            embedding = (
                _pil_to_embedding(image, self.embedding_size)
                if compute_embeddings
                else np.empty((0,), dtype=np.float32)
            )
            frames.append(
                SampledFrame(
                    index=int(frame_index),
                    timestamp=timestamp,
                    image=image,
                    embedding=embedding,
                )
            )

        metadata.setdefault("fps", raw_fps)
        metadata.setdefault("frames_indices", frame_indices)
        metadata.setdefault("total_num_frames", frame_count)
        metadata.setdefault("video_backend", "qwen_fetch_video")
        return SampledVideo(frames=frames, video=video, metadata=metadata)

    def load_streamingbench_artifact(self, artifact_path: str) -> SampledVideo:
        """Load a producer-resized StreamingBench artifact without decoding again."""

        absolute_path = os.path.abspath(artifact_path)
        metadata = load_decoded_cache_metadata(absolute_path)
        if metadata.get("artifact_format") != "qwen_fetch_video":
            raise ValueError(
                "Qwen StreamingBench requires a qwen_fetch_video artifact: "
                f"{absolute_path}"
            )
        with np.load(absolute_path, allow_pickle=False) as payload:
            frames_array = payload["frames"]
            source_ids = payload["source_ids"]
        if (
            frames_array.dtype != np.uint8
            or frames_array.ndim != 4
            or frames_array.shape[-1] != 3
            or len(frames_array) == 0
        ):
            raise ValueError(
                "Qwen StreamingBench artifact frames must be non-empty NHWC uint8; "
                f"got shape={frames_array.shape}, dtype={frames_array.dtype}"
            )
        source_ids = np.asarray(source_ids).reshape(-1)
        if len(source_ids) != len(frames_array):
            raise ValueError("Qwen StreamingBench artifact source_ids length mismatch")
        qwen_metadata = metadata.get("qwen_video_metadata")
        if not isinstance(qwen_metadata, dict):
            raise ValueError("Qwen StreamingBench artifact is missing video metadata")
        raw_fps = max(float(qwen_metadata.get("fps", metadata.get("avg_fps", 1.0)) or 1.0), 1e-6)
        start_time = float(metadata.get("start_time", 0.0) or 0.0)
        base_frame_index = int(round(start_time * raw_fps))
        compute_embeddings = self.config.get("compute_frame_embeddings", True)
        if isinstance(compute_embeddings, str):
            compute_embeddings = compute_embeddings.lower() in {
                "1", "true", "yes", "y", "on"
            }
        sampled_frames: List[SampledFrame] = []
        for source_id, array in zip(source_ids.tolist(), frames_array):
            local_index = int(source_id)
            image = Image.fromarray(array).convert("RGB")
            sampled_frames.append(
                SampledFrame(
                    index=base_frame_index + local_index,
                    timestamp=start_time + float(local_index) / raw_fps,
                    image=image,
                    embedding=(
                        _pil_to_embedding(image, self.embedding_size)
                        if compute_embeddings
                        else np.empty((0,), dtype=np.float32)
                    ),
                )
            )
        video = torch.from_numpy(frames_array).permute(0, 3, 1, 2).contiguous()
        processor_metadata = dict(qwen_metadata)
        processor_metadata["frames_indices"] = [int(value) for value in source_ids]
        processor_metadata["stream_absolute_start_time"] = start_time
        processor_metadata["stream_absolute_frame_indices"] = [
            frame.index for frame in sampled_frames
        ]
        return SampledVideo(
            frames=sampled_frames,
            video=video,
            metadata=processor_metadata,
        )

    def load_sampled_video(self, video_path: str) -> SampledVideo:
        absolute_path = os.path.abspath(video_path)
        if absolute_path not in self._fetch_video_cache_remaining_loads:
            return self._decode_video_with_fetch_video(video_path)
        with self._fetch_video_reuse_lock:
            cached = self._fetch_video_reuse_cache.get(absolute_path)
            if cached is not None:
                remaining = self._fetch_video_cache_remaining_loads[absolute_path] - 1
                if remaining < 0:
                    raise RuntimeError(
                        f"Planned fetch-video cache use count was exceeded: {absolute_path}"
                    )
                self._fetch_video_cache_remaining_loads[absolute_path] = remaining
                if remaining == 0:
                    del self._fetch_video_reuse_cache[absolute_path]
                self._fetch_video_loaded_paths.add(absolute_path)
                return cached

        sample = self._decode_video_with_fetch_video(video_path)
        with self._fetch_video_reuse_lock:
            remaining = self._fetch_video_cache_remaining_loads[absolute_path] - 1
            if remaining < 0:
                raise RuntimeError(
                    f"Planned fetch-video cache use count was exceeded: {absolute_path}"
                )
            self._fetch_video_cache_remaining_loads[absolute_path] = remaining
            if remaining > 0:
                self._fetch_video_reuse_cache[absolute_path] = sample
            else:
                self._fetch_video_reuse_cache.pop(absolute_path, None)
            self._fetch_video_loaded_paths.add(absolute_path)
        return sample

    def load_sampled_frames(self, video_path: str) -> List[SampledFrame]:
        return self.load_sampled_video(video_path).frames

    def load_sampled_video_since(self, video_path: str, start_time: float) -> SampledVideo:
        eps = 1e-6
        sample = self._decode_video_with_fetch_video(video_path, video_start=float(start_time) + eps)
        keep = [i for i, frame in enumerate(sample.frames) if frame.timestamp > float(start_time) + eps]
        if len(keep) == len(sample.frames):
            return sample
        if not keep:
            return SampledVideo(
                frames=[],
                video=sample.video[:0],
                metadata=dict(sample.metadata),
            )
        keep_tensor = torch.tensor(keep, dtype=torch.long, device=sample.video.device)
        metadata = dict(sample.metadata)
        metadata["frames_indices"] = [sample.frames[i].index for i in keep]
        return SampledVideo(
            frames=[sample.frames[i] for i in keep],
            video=sample.video.index_select(0, keep_tensor),
            metadata=metadata,
        )

    def load_sampled_frames_since(self, video_path: str, start_time: float) -> List[SampledFrame]:
        """Stable prefix-chunk sampler for open-window streaming.

        OVO chunked videos are prefixes ending at the question timestamp.  For a
        persistent video session we should append only frames whose timestamp is
        newer than the last prefix we already consumed.  Unlike the stateless
        path, this deliberately avoids a uniform max-frame resample because that
        would change old frame identities every time the prefix becomes longer.
        """

        return self.load_sampled_video_since(video_path, start_time).frames

    def _cluster_history(self, frames: Sequence[SampledFrame]) -> List[ClusterCenter]:
        clusters: List[ClusterCenter] = []
        for frame in frames:
            if not clusters:
                clusters.append(ClusterCenter.from_frame(frame, self.center_image_size))
                continue

            latest = clusters[-1]
            similarity = _cosine(latest.centroid, frame.embedding)
            if similarity < self.cluster_threshold:
                clusters.append(ClusterCenter.from_frame(frame, self.center_image_size))
            else:
                latest.merge(frame, self.center_image_size)
        return clusters

    def _retrieve_long_clusters(
        self,
        clusters: Sequence[ClusterCenter],
        recent: Sequence[SampledFrame],
    ) -> List[ClusterCenter]:
        if self.long_topk <= 0 or not clusters:
            return []

        if recent:
            probe = _l2_normalize(np.mean([f.embedding for f in recent], axis=0))
            scored = [(_cosine(cluster.centroid, probe), i, cluster) for i, cluster in enumerate(clusters)]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = [item[2] for item in scored[: self.long_topk]]
        else:
            selected = list(clusters[-self.long_topk :])

        selected.sort(key=lambda cluster: cluster.start_index)
        return selected

    def build_memory_frames(self, video_path: str) -> Tuple[List[Image.Image], Dict[str, object]]:
        frames = self.load_sampled_frames(video_path)
        if not frames:
            raise ValueError(f"No frames decoded from video: {video_path}")
        for frame in frames:
            if frame.embedding.size == 0:
                frame.embedding = _pil_to_embedding(frame.image, self.embedding_size)

        recent_n = max(self.recent_frames, 0)
        if recent_n > 0:
            history = frames[:-recent_n]
            recent = frames[-recent_n:]
        else:
            history = frames
            recent = []

        clusters = self._cluster_history(history)
        selected_clusters = self._retrieve_long_clusters(clusters, recent)

        # qwen-vl-utils stacks image-list video frames before its final video
        # resize step, so all frames must have identical H/W here.
        memory_images = [_resize_memory_image(cluster.to_image(), self.center_image_size) for cluster in selected_clusters]
        memory_images.extend([_resize_memory_image(frame.image, self.center_image_size) for frame in recent])

        if len(memory_images) == 1:
            memory_images.append(memory_images[0].copy())

        stats = {
            "sampled_frames": len(frames),
            "history_frames": len(history),
            "recent_frames": len(recent),
            "cluster_count": len(clusters),
            "selected_long_clusters": len(selected_clusters),
            "cluster_frame_counts": [cluster.count for cluster in clusters],
            "selected_cluster_ranges": [
                {
                    "start_time": cluster.start_time,
                    "end_time": cluster.end_time,
                    "count": cluster.count,
                }
                for cluster in selected_clusters
            ],
            "input_frames": len(memory_images),
        }
        return memory_images, stats
