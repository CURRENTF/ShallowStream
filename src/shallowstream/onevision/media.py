from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from transformers import (
    AutoConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from src.config import apply_config_sources
from src.shallowstream.common import (
    SingleLayerLegacyCache,
    expand_temporal_neighbors,
    get_active_attn_implementation,
)
from src.utils.eval_io import atomic_write_json
from src.utils.streamingbench_decode_cache import (
    decoded_cache_audio_source,
    load_decoded_cache_metadata,
)
from src.utils.time_trace import (
    StageRecorder,
    extract_clip_range_seconds,
    extract_sample_id,
    make_prompt_preview,
)

from .config import (
    LONG_CLUSTER_COSINE_SIM_THRESHOLD,
    NO_NEW_VIDEO_CHUNK,
    ONEVISION_V3_DEFAULT_CONFIG,
    new_runtime_state,
)


@dataclass(frozen=True)
class PreparedOneVisionVideo:
    """CPU-only decoded frames safe to prepare outside the runtime thread."""

    video_path: str
    frames: np.ndarray
    source_ids: Tuple[int, ...]
    sampled_fps: Optional[float]
    video_duration_s: Optional[float]
    pixel_values_batches: Tuple[torch.Tensor, ...] = ()


def load_prepared_onevision_video(video_path: str) -> PreparedOneVisionVideo:
    """Load an NPZ queue artifact without mutating OneVision runtime state."""

    absolute_path = os.path.abspath(str(video_path))
    if not absolute_path.lower().endswith(".npz"):
        raise ValueError(f"OneVision input prefetch requires an NPZ cache: {absolute_path}")
    metadata = load_decoded_cache_metadata(absolute_path)
    with np.load(absolute_path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frames"])
        source_ids = tuple(
            int(value)
            for value in np.asarray(payload["source_ids"]).astype(np.int64, copy=False)
        )
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"Invalid decoded frame tensor in cache: {absolute_path}")
    if len(source_ids) != len(frames):
        raise ValueError(
            f"Decoded cache source-id count mismatch: path={absolute_path}, "
            f"frames={len(frames)}, source_ids={len(source_ids)}"
        )
    avg_fps = metadata.get("avg_fps")
    duration = metadata.get("video_duration_s")
    return PreparedOneVisionVideo(
        video_path=absolute_path,
        frames=frames,
        source_ids=source_ids,
        sampled_fps=None if avg_fps is None else float(avg_fps),
        video_duration_s=None if duration is None else float(duration),
    )


def _apply_input_recent_window(
    config: Dict[str, Any],
    frames: np.ndarray,
    source_ids: Tuple[int, ...] | List[int],
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    recent = config.get("input_recent_frames")
    ids = tuple(int(value) for value in source_ids)
    if recent is None:
        return frames, ids
    window = int(recent)
    if window <= 0:
        raise ValueError(f"input_recent_frames must be positive, got {recent}")
    if len(frames) != len(ids):
        raise ValueError(
            "OneVision frame/source-id count mismatch before input window: "
            f"frames={len(frames)}, source_ids={len(ids)}"
        )
    return frames[-window:], ids[-window:]


class OneVisionMediaMixin:

    def prepare_video_frames(self, video_path: str) -> PreparedOneVisionVideo:
        prepared = load_prepared_onevision_video(video_path)
        frames, source_ids = _apply_input_recent_window(
            getattr(self, "config", {}),
            prepared.frames,
            prepared.source_ids,
        )
        prepared = replace(
            prepared,
            frames=frames,
            source_ids=source_ids,
        )
        batch_size = max(1, int(self.config.get("vision_batch_size", 1)))
        pixel_values_batches = []
        for start in range(0, len(prepared.frames), batch_size):
            frames = prepared.frames[start : start + batch_size]
            video_inputs = self.processor.video_processor(frames, return_tensors="pt")
            pixel_values = getattr(video_inputs, "pixel_values_videos", None)
            if pixel_values is None:
                raise RuntimeError("video_processor did not return pixel_values_videos")
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
            if pixel_values.ndim != 5:
                raise RuntimeError(
                    f"Unexpected prefetched pixel_values_videos ndim={pixel_values.ndim}"
                )
            if pixel_values.is_cuda:
                raise RuntimeError("OneVision input prefetch must keep visual inputs on CPU")
            pixel_values_batches.append(pixel_values)
        return replace(
            prepared,
            pixel_values_batches=tuple(pixel_values_batches),
        )

    def _extract_choice_letter(self, text: str) -> str:
        if not text:
            return ""
        tail = str(text)
        for marker in ["assistant", "The best option is", "best option is", "Answer:", "answer is"]:
            pos = tail.lower().rfind(marker.lower())
            if pos != -1:
                tail = tail[pos + len(marker):]
                break
        matches = re.findall(r"\b([ABCD])\b", tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        matches = re.findall(r"([ABCD])", tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        return ""

    def _decode_generated(self, output_ids: torch.Tensor, input_len: int) -> str:
        if output_ids is None:
            return ""
        gen_ids = output_ids[:, input_len:] if output_ids.shape[1] >= input_len else output_ids
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    def _infer_clip_start_seconds(self, video_path: str) -> float:
        if str(video_path).lower().endswith(".npz"):
            try:
                return float(load_decoded_cache_metadata(video_path).get("start_time", 0.0))
            except (OSError, ValueError, TypeError):
                return 0.0
        base = os.path.basename(video_path or "")
        m = re.search(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.(?:mp4|npz)$", base)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
        return 0.0

    def _resolve_existing_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        raw = str(path)
        candidates = []
        if os.path.isabs(raw):
            candidates.append(raw)
        else:
            stripped = raw.lstrip("./\\")
            candidates.extend(
                [
                    os.path.abspath(raw),
                    os.path.abspath(stripped),
                    os.path.abspath(os.path.join("data", stripped)),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", stripped)),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", stripped)),
                ]
            )
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    def _pick_audio_source(self, video_path: str, audio_path: Optional[str]) -> Optional[str]:
        mode = str(self.config.get("audio_source", "video")).lower()
        video_source = decoded_cache_audio_source(video_path) or self._resolve_existing_path(video_path)
        audio_source = self._resolve_existing_path(audio_path)
        if mode == "audio":
            return audio_source
        if mode == "audio_or_video":
            return audio_source or video_source
        return video_source

    def _audio_cache_path(self, source_path: str, clip_start_s: float) -> str:
        cache_dir = os.path.abspath(str(self.config.get("asr_cache_dir", "./outputs/streamingbench/cache/asr")))
        os.makedirs(cache_dir, exist_ok=True)
        try:
            stat = os.stat(source_path)
            signature = f"{os.path.abspath(source_path)}|{stat.st_size}|{stat.st_mtime_ns}|{clip_start_s:.3f}"
        except OSError:
            signature = f"{os.path.abspath(source_path)}|missing|{clip_start_s:.3f}"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        return os.path.join(cache_dir, f"{digest}.json")

    def _ensure_asr_model_loaded(self):
        if self._asr_model is not None:
            return self._asr_model
        if self._asr_unavailable_reason is not None:
            raise RuntimeError(self._asr_unavailable_reason)
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            self._asr_unavailable_reason = (
                "faster-whisper is not installed. Install it with: pip install -U faster-whisper"
            )
            raise RuntimeError(self._asr_unavailable_reason) from exc

        device_cfg = str(self.config.get("asr_device", "auto")).lower()
        if device_cfg == "auto":
            asr_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            asr_device = device_cfg
        compute_type = str(self.config.get("asr_compute_type", "float16"))
        if asr_device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        try:
            self._asr_model = WhisperModel(
                str(self.config.get("asr_model_name", "small")),
                device=asr_device,
                compute_type=compute_type,
            )
            print(
                f"[{self.log_name}-Audio] ASR loaded: model={self.config.get('asr_model_name')} "
                f"device={asr_device} compute_type={compute_type}",
                flush=True,
            )
        except Exception as exc:
            self._asr_unavailable_reason = f"failed to load ASR model: {exc}"
            raise RuntimeError(self._asr_unavailable_reason) from exc
        return self._asr_model

    def _transcribe_audio_segments(self, source_path: str, clip_start_s: float) -> List[Dict[str, Any]]:
        if not self.config.get("enable_audio_transcript"):
            return []
        cache_path = self._audio_cache_path(source_path, clip_start_s)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                segments = cached.get("segments", [])
                if isinstance(segments, list):
                    return segments
            except Exception as exc:
                error = RuntimeError(f"Invalid ASR cache: {cache_path}")
                if self.explicit_asr_cache_cause:
                    raise error from exc
                raise error

        asr = self._ensure_asr_model_loaded()
        try:
            segments_iter, _ = asr.transcribe(
                source_path,
                beam_size=int(self.config.get("asr_beam_size", 5)),
                vad_filter=bool(self.config.get("asr_vad_filter", True)),
            )
            segments: List[Dict[str, Any]] = []
            for seg in segments_iter:
                text = str(getattr(seg, "text", "")).strip()
                if not text:
                    continue
                start = clip_start_s + float(getattr(seg, "start", 0.0))
                end = clip_start_s + float(getattr(seg, "end", start))
                segments.append({"start": start, "end": max(end, start), "text": text})
            atomic_write_json(
                cache_path,
                {
                    "source": os.path.abspath(source_path),
                    "clip_start_s": clip_start_s,
                    "segments": segments,
                },
                indent=2,
            )
            self._dbg(f"audio_transcribe: source={source_path} segments={len(segments)} cache={cache_path}")
            return segments
        except Exception as exc:
            raise RuntimeError(f"ASR transcription failed for {source_path}: {exc}") from exc

    def _merge_audio_segments(self, new_segments: List[Dict[str, Any]]) -> None:
        if not new_segments:
            return
        existing = self.state.get("audio_segments")
        if not isinstance(existing, list):
            existing = []
        merged = existing + new_segments
        seen = set()
        unique: List[Dict[str, Any]] = []
        for seg in sorted(merged, key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0)))):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            key = (round(float(seg.get("start", 0.0)), 2), round(float(seg.get("end", 0.0)), 2), text)
            if key in seen:
                continue
            seen.add(key)
            unique.append({"start": key[0], "end": key[1], "text": text})
        self.state["audio_segments"] = unique

    def _update_audio_segments_for_chunk(self, video_path: str, audio_path: Optional[str], clip_start_s: float) -> None:
        if not self.config.get("enable_audio_transcript"):
            return
        source = self._pick_audio_source(video_path, audio_path)
        if source is None:
            self._dbg("audio_transcribe: no usable audio/video source found")
            return
        self._merge_audio_segments(self._transcribe_audio_segments(source, clip_start_s))

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minute = int(seconds // 60)
        sec = seconds - minute * 60
        return f"{minute:02d}:{sec:04.1f}"

    def _segments_for_frame_ts(self, frame_ts: float) -> List[Dict[str, Any]]:
        segments = self.state.get("audio_segments")
        if not isinstance(segments, list) or not segments:
            return []
        radius = max(0.0, float(self.config.get("audio_frame_context_radius_s", 1.25)))
        start = float(frame_ts) - radius
        end = float(frame_ts) + radius
        matched = [
            seg
            for seg in segments
            if float(seg.get("end", 0.0)) >= start and float(seg.get("start", 0.0)) <= end
        ]
        if matched:
            return matched

        nearest_radius = max(0.0, float(self.config.get("audio_nearest_segment_radius_s", 2.5)))
        nearest: List[Tuple[float, Dict[str, Any]]] = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            if seg_start <= frame_ts <= seg_end:
                distance = 0.0
            else:
                distance = min(abs(frame_ts - seg_start), abs(frame_ts - seg_end))
            if distance <= nearest_radius:
                nearest.append((distance, seg))
        nearest.sort(key=lambda x: x[0])
        return [seg for _, seg in nearest[:2]]

    def _caption_text_for_frame_ts(self, frame_ts: float) -> str:
        pieces = []
        seen_text = set()
        for seg in self._segments_for_frame_ts(float(frame_ts)):
            text = str(seg.get("text", "")).strip()
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            pieces.append(
                f"[{self._format_seconds(float(seg.get('start', 0.0)))}-"
                f"{self._format_seconds(float(seg.get('end', 0.0)))}] {text}"
            )
        return " ".join(pieces).strip()

    def _caption_token_ids_for_frame(self, frame_ts: float, caption: str, device: str) -> Optional[torch.Tensor]:
        caption = str(caption or "").strip()
        if not caption:
            return None
        text = f"\nAudio transcript near frame {self._format_seconds(float(frame_ts))}: {caption}\n"
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        max_tokens = max(0, int(self.config.get("audio_caption_max_tokens_per_frame", 64)))
        if max_tokens <= 0:
            return None
        if ids.shape[1] > max_tokens:
            ids = ids[:, :max_tokens].contiguous()
        return ids if ids.numel() > 0 else None

    def _caption_token_ids_batch_for_frames(
        self,
        frame_ts_list: List[float],
        captions: List[str],
        device: str,
    ) -> List[Optional[torch.Tensor]]:
        total = min(len(frame_ts_list), len(captions))
        out: List[Optional[torch.Tensor]] = [None] * total
        max_tokens = max(0, int(self.config.get("audio_caption_max_tokens_per_frame", 64)))
        if total <= 0 or max_tokens <= 0:
            return out

        text_inputs: List[str] = []
        row_to_frame_idx: List[int] = []
        for i in range(total):
            caption = str(captions[i] if i < len(captions) else "").strip()
            if not caption:
                continue
            ts = float(frame_ts_list[i])
            text_inputs.append(f"\nAudio transcript near frame {self._format_seconds(ts)}: {caption}\n")
            row_to_frame_idx.append(i)

        if not text_inputs:
            return out

        encoded = self.tokenizer(
            text_inputs,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        input_ids_batch = encoded.get("input_ids", [])
        if not isinstance(input_ids_batch, list):
            return out

        for row_idx, frame_idx in enumerate(row_to_frame_idx):
            if row_idx >= len(input_ids_batch):
                break
            token_ids = input_ids_batch[row_idx]
            if not isinstance(token_ids, list) or len(token_ids) == 0:
                continue
            out[frame_idx] = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        return out

    def _refresh_frame_captions_from_audio(self) -> None:
        frame_ts = self.state.get("frame_source_ids")
        if not isinstance(frame_ts, list):
            self.state["frame_captions"] = []
            return
        self.state["frame_captions"] = [self._caption_text_for_frame_ts(float(ts)) for ts in frame_ts]

    def _build_selected_audio_context(self, selected_frames: List[int]) -> str:
        if not self.config.get("enable_audio_transcript"):
            return ""
        captions = self.state.get("frame_captions")
        source_ts = self.state.get("frame_source_ids")
        if not isinstance(captions, list) or not isinstance(source_ts, list):
            return ""
        total = min(len(captions), len(source_ts))
        lines: List[str] = []
        seen = set()
        for fid in sorted(set(int(x) for x in selected_frames if 0 <= int(x) < total)):
            caption = str(captions[fid]).strip()
            if not caption:
                continue
            key = (round(float(source_ts[fid]), 2), caption)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- frame {fid} @ {self._format_seconds(float(source_ts[fid]))}: {caption}")
        if not lines:
            return ""
        context = str(self.config.get("audio_context_prefix", "")).strip()
        context = context + "\n" + "\n".join(lines) if context else "\n".join(lines)
        max_chars = max(0, int(self.config.get("max_audio_context_chars", 1200)))
        if max_chars > 0 and len(context) > max_chars:
            context = context[:max_chars].rsplit("\n", 1)[0].strip()
        return context.strip()

    def _load_video_frames(
        self,
        video_path: str,
        sample_fps: float,
        max_frames_num: Optional[int],
        prepared_video: Optional[PreparedOneVisionVideo] = None,
    ) -> Tuple[np.ndarray, List[int], Optional[float]]:
        self.state["_last_video_duration_s"] = None
        if video_path is None:
            raise ValueError("video_path is None; upstream video split likely failed")
        if prepared_video is not None:
            requested_path = os.path.abspath(str(video_path))
            if prepared_video.video_path != requested_path:
                raise ValueError(
                    "Prepared OneVision video does not match requested path: "
                    f"prepared={prepared_video.video_path}, requested={requested_path}"
                )
            self.state["_last_video_duration_s"] = prepared_video.video_duration_s
            frames, source_ids = _apply_input_recent_window(
                getattr(self, "config", {}),
                prepared_video.frames,
                prepared_video.source_ids,
            )
            return frames, list(source_ids), prepared_video.sampled_fps
        if str(video_path).lower().endswith(".npz"):
            prepared = load_prepared_onevision_video(video_path)
            self.state["_last_video_duration_s"] = prepared.video_duration_s
            frames, source_ids = _apply_input_recent_window(
                getattr(self, "config", {}),
                prepared.frames,
                prepared.source_ids,
            )
            return frames, list(source_ids), prepared.sampled_fps
        if video_path.endswith((".jpg", ".jpeg", ".png")):
            image = np.asarray(Image.open(video_path).convert("RGB"))
            frames, source_ids = _apply_input_recent_window(
                getattr(self, "config", {}),
                np.expand_dims(image, axis=0),
                (0,),
            )
            return frames, list(source_ids), None

        if os.path.isdir(video_path):
            frame_files = sorted(
                [
                    os.path.join(video_path, name)
                    for name in os.listdir(video_path)
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if not frame_files:
                raise ValueError(f"No image frames found in directory: {video_path}")
            frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_files]
            video = np.stack(frames, axis=0)
            src_idx = list(range(len(video)))
            if max_frames_num is not None and len(video) > max_frames_num:
                idx = np.linspace(0, len(video) - 1, max_frames_num, dtype=int)
                video = video[idx]
                src_idx = [src_idx[int(i)] for i in idx.tolist()]
            video, source_ids = _apply_input_recent_window(
                getattr(self, "config", {}),
                video,
                src_idx,
            )
            return video, list(source_ids), None

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=10)
        total_frame_num = len(vr)
        if total_frame_num == 0:
            raise ValueError(f"Empty video: {video_path}")
        avg_fps = float(vr.get_avg_fps()) if float(vr.get_avg_fps()) > 0 else 1.0
        self.state["_last_video_duration_s"] = float(total_frame_num) / float(avg_fps)

        if max_frames_num is not None:
            frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        else:
            fps = max(round(avg_fps), 1)
            step = max(int(round(fps / sample_fps)), 1)
            frame_idx = list(range(0, total_frame_num, step))

        frames, source_ids = _apply_input_recent_window(
            getattr(self, "config", {}),
            vr.get_batch(frame_idx).asnumpy(),
            frame_idx,
        )
        return frames, list(source_ids), avg_fps
