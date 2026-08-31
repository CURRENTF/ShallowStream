from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
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
from src.shallowstream.evidence_retrieval import evidence_retrieval_backend
from src.utils.eval_io import atomic_write_json
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


class OneVisionPrefillMixin:
    def _lower_cache_seq_len(self, lower_kv: Optional[Dict[int, Dict[str, torch.Tensor]]]) -> int:
        if not isinstance(lower_kv, dict) or not lower_kv:
            return 0
        for entry in lower_kv.values():
            if isinstance(entry, dict) and isinstance(entry.get("k"), torch.Tensor):
                return int(entry["k"].shape[-2])
        return 0

    def _retain_frame_hidden_for_prefill(self) -> bool:
        if bool(self.config.get("debug_similarity", False)):
            return True
        if bool(self.config.get("full_kv_mode", False)):
            return False
        redundant_for_token_vote_prompt = bool(
            self._selected_generate_mode() == "simple_prompt"
            and self._retrieval_score_strategy() == "shallow_layer_token_vote"
            and evidence_retrieval_backend(self.config) == "shallow"
            and int(self.config.get("long_cluster_topk", 0)) <= 0
        )
        return not redundant_for_token_vote_prompt

    def _trim_lower_kv_to_window(self) -> None:
        if bool(self.config.get("full_kv_mode", False)):
            return
        if int(self.config.get("retrieval_search_last_n_frames", 0)) <= 0:
            return

        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if not isinstance(lower_kv, dict) or not lower_kv:
            return
        if not isinstance(frame_spans, list):
            frame_spans = []

        seq_len = self._lower_cache_seq_len(lower_kv)
        init_len = max(0, min(int(self.state.get("init_len", 0)), seq_len))
        local_frames = int(self.config.get("shallow_prefill_local_window_frames", 0))
        tokens_per_frame = int(self.config.get("n_frame_tokens", 0))
        if local_frames <= 0 or tokens_per_frame <= 0:
            raise ValueError(
                "OneVision LC cache eviction requires positive local-window frames "
                "and frame-token count"
            )

        # Future lower-layer appends can only attend to the sink plus this exact
        # token tail. Recent retrieval spans may extend farther back because
        # captions make frame sizes variable, so retain their complete union too.
        tail_start = max(init_len, seq_len - local_frames * tokens_per_frame)
        recent_starts: List[int] = []
        for span in frame_spans:
            if not isinstance(span, dict):
                continue
            starts = [
                int(span.get("visual_start", -1)),
                int(span.get("caption_start", -1)),
            ]
            recent_starts.extend(x for x in starts if x >= init_len)
        if recent_starts:
            tail_start = min(tail_start, min(recent_starts))
        if tail_start <= init_len:
            return

        keep_idx = torch.cat(
            [
                torch.arange(init_len, dtype=torch.long),
                torch.arange(tail_start, seq_len, dtype=torch.long),
            ],
            dim=0,
        )
        compacted: Dict[int, Dict[str, torch.Tensor]] = {}
        for lid, entry in lower_kv.items():
            if not isinstance(entry, dict):
                raise RuntimeError(f"Invalid OneVision lower cache entry at layer {lid}")
            k = entry.get("k")
            v = entry.get("v")
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise RuntimeError(f"Missing OneVision lower cache K/V at layer {lid}")
            if int(k.shape[-2]) != seq_len or int(v.shape[-2]) != seq_len:
                raise RuntimeError(
                    "OneVision lower-cache layers are not sequence-aligned: "
                    f"layer={lid} k={int(k.shape[-2])} v={int(v.shape[-2])} expected={seq_len}"
                )
            pos_value = entry.get("pos")
            if isinstance(pos_value, torch.Tensor):
                pos = pos_value.to(device=k.device, dtype=torch.long).reshape(-1)
                if int(pos.numel()) != seq_len:
                    raise RuntimeError(
                        "Invalid OneVision lower-cache positions during eviction: "
                        f"layer={lid} got={int(pos.numel())} expected={seq_len}"
                    )
            else:
                pos = torch.arange(seq_len, device=k.device, dtype=torch.long)
            layer_idx = keep_idx.to(device=k.device)
            compacted[int(lid)] = {
                "k": k.index_select(dim=-2, index=layer_idx).detach(),
                "v": v.index_select(dim=-2, index=layer_idx).detach(),
                "pos": pos.index_select(dim=0, index=layer_idx).detach(),
            }

        def _remap_boundary(old: int) -> int:
            if old < init_len:
                return old
            if old < tail_start:
                raise RuntimeError(
                    "OneVision LC eviction would drop a retained frame span: "
                    f"boundary={old} tail_start={tail_start}"
                )
            return init_len + old - tail_start

        remapped_spans: List[Dict[str, Any]] = []
        for span in frame_spans:
            if not isinstance(span, dict):
                continue
            remapped = dict(span)
            for key in ("visual_start", "visual_end", "caption_start", "caption_end"):
                remapped[key] = _remap_boundary(int(span[key]))
            remapped_spans.append(remapped)

        self.state["lower_kv"] = compacted
        self.state["frame_spans"] = remapped_spans
        self._dbg(
            "trim_lower_kv: "
            f"old_tokens={seq_len} new_tokens={self._lower_cache_seq_len(compacted)} "
            f"dropped={tail_start - init_len} absolute_tail_start={tail_start}"
        )

    def _reset_stream_state(self, session_id: Optional[str], sample_fps: float, max_frames_num: Optional[int], device: str) -> None:
        self.state["session"] = session_id
        self.state["sample_fps"] = sample_fps
        self.state["max_frames_num"] = max_frames_num
        self.state["lower_kv"] = None
        self.state["init_len"] = 0
        self.state["init_input_embeds"] = None
        self.state["init_hidden_l8"] = None
        self.state["frame_input_embeds"] = None
        self.state["frame_source_images"] = []
        self.state["frame_hidden_l8"] = None
        self.state["frame_source_ids"] = []
        self.state["frame_spans"] = []
        self.state["frame_caption_hidden_l8"] = []
        self.state["frame_debug_thumbs"] = []
        self.state["frame_evidence_images"] = []
        self.state["audio_segments"] = []
        self.state["frame_captions"] = []
        self.state["question_counter"] = 0
        self.state["last_selection"] = {}
        self.state["last_gate_decision"] = {}
        self.state["last_latest_unit_score_observation"] = {}
        self.state["last_simple_prompt_stats"] = {}
        self.state["long_clusters"] = []
        self.state["lower_attn_path_stats"] = self._new_lower_attn_path_stats()
        self.state["upper_flash_layer_calls"] = 0
        self.state["upper_nonflash_layer_calls"] = 0

        init_prompt = "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user "
        init_ids = self._tokenize(init_prompt, device)
        layers, embed_tokens, _, _ = self._get_lm_components()
        hidden = embed_tokens(init_ids).to(dtype=torch.float16, device=device)

        with torch.inference_mode():
            hidden_l8, lower_kv = self._forward_lower_layers_raw(
                hidden_states=hidden,
                start_layer=0,
                end_layer=min(self.config["prune_layer"], len(layers)),
                past_raw_kv={},
                update_cache=True,
            )

        self.state["lower_kv"] = lower_kv
        self.state["init_len"] = init_ids.shape[1]
        self.state["init_input_embeds"] = hidden[0].detach().to(device="cpu")
        self.state["init_hidden_l8"] = hidden_l8[0].detach()
        self.state["frame_input_embeds"] = torch.empty(
            (0, self.config["n_frame_tokens"], hidden_l8.shape[-1]),
            device="cpu",
            dtype=hidden_l8.dtype,
        )
        self.state["frame_hidden_l8"] = torch.empty(
            (0, self.config["n_frame_tokens"], hidden_l8.shape[-1]),
            device=hidden_l8.device,
            dtype=hidden_l8.dtype,
        )
        self._dbg(f"reset_stream: init_len={self.state['init_len']}")

    def _append_video_chunk_prefill_lower(
        self,
        frames: np.ndarray,
        source_ids: List[float],
        device: str,
        prepared_video: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if frames.size == 0:
            return {
                "frame_count": 0,
                "feature_extract_ms": 0.0,
                "visual_prefill_ms": 0.0,
                "caption_tokenize_ms": 0.0,
                "caption_prefill_ms": 0.0,
                "caption_token_count": 0,
                "long_promoted_frames": 0,
                "long_cluster_count": int(len(self.state.get("long_clusters") or [])),
                "total_ms": 0.0,
            }

        t_total_start = time.perf_counter()
        t_feature_start = time.perf_counter()
        prepared_pixel_values = (
            prepared_video.pixel_values_batches
            if prepared_video is not None and prepared_video.pixel_values_batches
            else None
        )
        if prepared_pixel_values is None:
            video_feats = self._extract_video_features_rekv(frames, device)
        else:
            video_feats = self._extract_video_features_rekv(
                frames,
                device,
                prepared_pixel_values=prepared_pixel_values,
            )
        feature_extract_ms = (time.perf_counter() - t_feature_start) * 1000.0
        visual_feature_bytes = int(video_feats.numel() * video_feats.element_size())
        fullkv_visual_features_staged_on_cpu = bool(
            self.config.get("full_kv_mode", False)
            and torch.cuda.is_available()
            and video_feats.is_cuda
        )
        large_visual_features_staged_on_cpu = bool(
            torch.cuda.is_available()
            and video_feats.is_cuda
            and visual_feature_bytes >= 4 * 1024**3
        )
        if fullkv_visual_features_staged_on_cpu or large_visual_features_staged_on_cpu:
            # Large immutable visual features compete with growing KV for device
            # memory. Preserve their exact fp16 values on CPU and copy only each
            # consumed micro-batch back to the device.
            video_feats = video_feats.cpu()
            torch.cuda.empty_cache()
        nf, tpf, hd = video_feats.shape
        configured_tpf = int(self.config.get("n_frame_tokens", 0))
        existing_frames = self.state.get("frame_hidden_l8")
        if (
            isinstance(existing_frames, torch.Tensor)
            and existing_frames.numel() > 0
            and int(existing_frames.shape[1]) != int(tpf)
        ):
            raise RuntimeError(
                "OneVision visual token count changed within one stream: "
                f"existing={int(existing_frames.shape[1])} current={int(tpf)}"
            )
        if configured_tpf != int(tpf):
            self.config["n_frame_tokens"] = int(tpf)
            self._dbg(
                "resolved n_frame_tokens from vision output: "
                f"configured={configured_tpf} actual={int(tpf)}"
            )
        self._dbg(f"append_video_chunk: frames={nf} tokens={nf*tpf}")

        layers, embed_tokens, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        retain_frame_hidden = self._retain_frame_hidden_for_prefill()
        retain_frame_input = self._selected_generate_mode() == "simple_prompt"
        retain_frame_source_images = bool(
            retain_frame_input
            and (
                self.config.get("selected_prompt_reencode_all_as_images", False)
                or self.config.get(
                    "selected_prompt_reencode_recent_as_images",
                    False,
                )
            )
        )
        existing_source_images = self.state.get("frame_source_images")
        if not isinstance(existing_source_images, list):
            existing_source_images = []
        self.state["frame_source_images"] = existing_source_images
        if retain_frame_source_images:
            retained_input = self.state.get("frame_input_embeds")
            retained_frame_count = (
                int(retained_input.shape[0])
                if isinstance(retained_input, torch.Tensor)
                else 0
            )
            if len(existing_source_images) != retained_frame_count:
                raise RuntimeError(
                    "OneVision image-reencode source buffer is not aligned with short memory: "
                    f"images={len(existing_source_images)}, frames={retained_frame_count}"
                )
        else:
            self.state["frame_source_images"] = []
        new_frame_captions: List[str] = []
        new_frame_spans: List[Dict[str, Any]] = []
        new_source_ids: List[float] = []
        new_debug_thumbs: List[np.ndarray] = []
        new_evidence_images: List[Image.Image] = []
        lower_cache = self.state["lower_kv"]
        visual_prefill_ms = 0.0
        caption_tokenize_ms = 0.0
        caption_prefill_ms = 0.0
        caption_token_count = 0
        long_promoted_frames = 0
        long_cluster_count = int(len(self.state.get("long_clusters") or []))
        base_frame_id = 0
        if isinstance(self.state.get("frame_spans"), list):
            base_frame_id = len(self.state["frame_spans"])

        frame_ts_list: List[float] = []
        for i in range(nf):
            frame_ts = float(source_ids[i]) if i < len(source_ids) else 0.0
            frame_ts_list.append(frame_ts)
            new_source_ids.append(frame_ts)
            new_frame_captions.append(self._caption_text_for_frame_ts(frame_ts))
            if self.config.get("debug_similarity"):
                new_debug_thumbs.append(self._make_debug_thumb(frames[i]))
            if evidence_retrieval_backend(self.config) == "siglip":
                new_evidence_images.append(
                    Image.fromarray(frames[i]).convert("RGB").copy()
                )

        t_caption_tokenize_start = time.perf_counter()
        caption_ids_list = self._caption_token_ids_batch_for_frames(frame_ts_list, new_frame_captions, device)
        caption_tokenize_ms = (time.perf_counter() - t_caption_tokenize_start) * 1000.0

        with torch.inference_mode():
            batch_size = max(1, int(self.config.get("prefill_interleave_batch_size", 8)))
            i = 0
            while i < nf:
                j = min(nf, i + batch_size)

                # Build one interleaved token stream per micro-batch:
                # [frame_i_visual, frame_i_caption?, frame_{i+1}_visual, ...]
                stream_parts: List[torch.Tensor] = []
                segs: List[Dict[str, Any]] = []
                frame_items: List[Dict[str, Any]] = []
                batch_hidden_frames: List[torch.Tensor] = []
                batch_caption_hidden: List[torch.Tensor] = []
                batch_frame_spans: List[Dict[str, Any]] = []
                total_visual_tokens = 0
                total_caption_tokens = 0

                for k in range(i, j):
                    frame_ts = frame_ts_list[k]
                    caption = new_frame_captions[k]
                    frame_embed = video_feats[k : k + 1].reshape(1, tpf, hd).to(device=device)
                    stream_parts.append(frame_embed)
                    segs.append({"kind": "visual", "idx": k, "len": int(tpf)})
                    total_visual_tokens += int(tpf)

                    caption_ids = caption_ids_list[k] if k < len(caption_ids_list) else None
                    cap_len = 0
                    if caption_ids is not None and caption_ids.numel() > 0:
                        cap_len = int(caption_ids.shape[1])
                        caption_embed = embed_tokens(caption_ids).to(dtype=torch.float16, device=device)
                        stream_parts.append(caption_embed)
                        segs.append({"kind": "caption", "idx": k, "len": cap_len})
                        total_caption_tokens += cap_len

                    frame_items.append(
                        {
                            "idx": k,
                            "source_ts": frame_ts,
                            "caption": caption,
                            "visual_start": -1,
                            "visual_end": -1,
                            "caption_start": -1,
                            "caption_end": -1,
                            "caption_hidden": None,
                        }
                    )

                if not stream_parts:
                    break

                prefill_input = torch.cat(stream_parts, dim=1)
                stream_start = self._lower_cache_seq_len(lower_cache)
                t_prefill_start = time.perf_counter()
                hidden_l8_stream, lower_cache = self._forward_lower_layers_raw(
                    hidden_states=prefill_input,
                    start_layer=0,
                    end_layer=prune,
                    past_raw_kv=lower_cache,
                    update_cache=True,
                )
                prefill_elapsed_ms = (time.perf_counter() - t_prefill_start) * 1000.0

                # Keep the old timing fields for compatibility by splitting mixed
                # prefill time proportionally to visual/caption token counts.
                total_tokens = max(1, total_visual_tokens + total_caption_tokens)
                visual_prefill_ms += prefill_elapsed_ms * (float(total_visual_tokens) / float(total_tokens))
                caption_prefill_ms += prefill_elapsed_ms * (float(total_caption_tokens) / float(total_tokens))
                caption_token_count += int(total_caption_tokens)

                hidden_seq = hidden_l8_stream[0]
                hdim = int(hidden_seq.shape[-1])
                stream_cursor = 0
                pos_cursor = stream_start
                frame_by_idx = {int(item["idx"]): item for item in frame_items}

                for seg in segs:
                    seg_len = int(seg["len"])
                    seg_idx = int(seg["idx"])
                    item = frame_by_idx.get(seg_idx)
                    if item is None:
                        stream_cursor += seg_len
                        pos_cursor += seg_len
                        continue

                    seg_hidden = hidden_seq[stream_cursor : stream_cursor + seg_len]
                    seg_start = pos_cursor
                    seg_end = seg_start + seg_len

                    if seg["kind"] == "visual":
                        item["visual_start"] = seg_start
                        item["visual_end"] = seg_end
                        item["caption_start"] = seg_end
                        item["caption_end"] = seg_end
                        if retain_frame_hidden:
                            batch_hidden_frames.append(
                                seg_hidden.reshape(1, seg_len, hdim).detach()
                            )
                    else:
                        item["caption_start"] = seg_start
                        item["caption_end"] = seg_end
                        if retain_frame_hidden:
                            item["caption_hidden"] = seg_hidden.detach()

                    stream_cursor += seg_len
                    pos_cursor += seg_len

                for item in frame_items:
                    cap_hidden = item["caption_hidden"]
                    if not isinstance(cap_hidden, torch.Tensor):
                        cap_hidden = torch.empty((0, hdim), device=hidden_seq.device, dtype=hidden_seq.dtype)
                    batch_caption_hidden.append(cap_hidden)
                    span = {
                        "frame_id": base_frame_id + int(item["idx"]),
                        "source_ts": float(item["source_ts"]),
                        "visual_start": int(item["visual_start"]),
                        "visual_end": int(item["visual_end"]),
                        "caption_start": int(item["caption_start"]),
                        "caption_end": int(item["caption_end"]),
                        "caption": str(item["caption"]),
                    }
                    batch_frame_spans.append(span)
                    new_frame_spans.append(dict(span))

                # Commit each micro-batch before processing the next one. This
                # lets long-cluster promotion and raw-KV eviction happen inside
                # a long chunk instead of only after the entire chunk is cached.
                self.state["lower_kv"] = lower_cache
                if retain_frame_input:
                    frame_inputs = video_feats[i:j].detach().to(device="cpu")
                    existing_inputs = self.state.get("frame_input_embeds")
                    if not isinstance(existing_inputs, torch.Tensor) or existing_inputs.numel() == 0:
                        self.state["frame_input_embeds"] = frame_inputs
                    else:
                        self.state["frame_input_embeds"] = torch.cat(
                            [existing_inputs, frame_inputs],
                            dim=0,
                        )
                if retain_frame_source_images:
                    self.state["frame_source_images"].extend(
                        Image.fromarray(frame).convert("RGB").copy()
                        for frame in frames[i:j]
                    )

                hidden_frames = (
                    torch.cat(batch_hidden_frames, dim=0)
                    if batch_hidden_frames
                    else torch.empty(
                        (0, tpf, hd),
                        device=device,
                        dtype=video_feats.dtype,
                    )
                )
                existing_hidden = self.state.get("frame_hidden_l8")
                if not isinstance(existing_hidden, torch.Tensor) or existing_hidden.numel() == 0:
                    self.state["frame_hidden_l8"] = hidden_frames
                elif hidden_frames.numel() > 0:
                    self.state["frame_hidden_l8"] = torch.cat(
                        [existing_hidden, hidden_frames],
                        dim=0,
                    )

                if not isinstance(self.state.get("frame_source_ids"), list):
                    self.state["frame_source_ids"] = []
                self.state["frame_source_ids"].extend(new_source_ids[i:j])
                if not isinstance(self.state.get("frame_captions"), list):
                    self.state["frame_captions"] = []
                self.state["frame_captions"].extend(new_frame_captions[i:j])
                if not isinstance(self.state.get("frame_caption_hidden_l8"), list):
                    self.state["frame_caption_hidden_l8"] = []
                self.state["frame_caption_hidden_l8"].extend(batch_caption_hidden)
                if not isinstance(self.state.get("frame_spans"), list):
                    self.state["frame_spans"] = []
                self.state["frame_spans"].extend(batch_frame_spans)
                if self.config.get("debug_similarity"):
                    if not isinstance(self.state.get("frame_debug_thumbs"), list):
                        self.state["frame_debug_thumbs"] = []
                    self.state["frame_debug_thumbs"].extend(new_debug_thumbs[i:j])
                if new_evidence_images:
                    if not isinstance(self.state.get("frame_evidence_images"), list):
                        self.state["frame_evidence_images"] = []
                    self.state["frame_evidence_images"].extend(new_evidence_images[i:j])

                long_update_stats = self._update_long_clusters_from_history(device=device)
                long_promoted_frames += int(long_update_stats.get("promoted_frames", 0))
                long_cluster_count = int(
                    long_update_stats.get(
                        "long_cluster_count",
                        len(self.state.get("long_clusters") or []),
                    )
                )
                self._trim_lower_kv_to_window()
                lower_cache = self.state["lower_kv"]

                i = j
                if bool(self.config.get("full_kv_mode", False)) and torch.cuda.is_available():
                    # Reclaim only inactive temporary blocks between equivalent
                    # FullKV micro-batches. Live KV tensors remain referenced.
                    torch.cuda.empty_cache()

        self._dbg_frames(
            "prefill_spans="
            + str(
                [
                    {
                        "frame": s["frame_id"],
                        "ts": round(float(s["source_ts"]), 3),
                        "visual": [s["visual_start"], s["visual_end"]],
                        "caption": [s["caption_start"], s["caption_end"]],
                        "caption_tokens": int(s["caption_end"]) - int(s["caption_start"]),
                    }
                    for s in new_frame_spans
                ]
            )
        )

        # Help release transient tensors earlier in long video streams.
        del video_feats
        return {
            "frame_count": int(nf),
            "feature_extract_ms": float(feature_extract_ms),
            "visual_prefill_ms": float(visual_prefill_ms),
            "caption_tokenize_ms": float(caption_tokenize_ms),
            "caption_prefill_ms": float(caption_prefill_ms),
            "caption_token_count": int(caption_token_count),
            "fullkv_visual_features_staged_on_cpu": fullkv_visual_features_staged_on_cpu,
            "large_visual_features_staged_on_cpu": large_visual_features_staged_on_cpu,
            "visual_feature_bytes": visual_feature_bytes,
            "fullkv_unused_frame_hidden_discarded": not retain_frame_hidden,
            "long_promoted_frames": int(long_promoted_frames),
            "long_cluster_count": int(long_cluster_count),
            "total_ms": float((time.perf_counter() - t_total_start) * 1000.0),
        }
