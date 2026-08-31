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


class OneVisionDecodeMixin:
    def _build_selected_split_cache(
        self,
        selected_frames: List[int],
        device: str,
        source_lower_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        append_question_tokens: int = 0,
        selected_long_cluster_ids: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        lower_kv = source_lower_kv if source_lower_kv is not None else self.state.get("lower_kv")
        frame_hidden = self.state.get("frame_hidden_l8")
        frame_spans = self.state.get("frame_spans")
        long_clusters = self.state.get("long_clusters")
        init_len = int(self.state.get("init_len", 0))
        if lower_kv is None:
            return {}, {}
        total_frames = 0 if frame_hidden is None else int(frame_hidden.shape[0])
        if not isinstance(frame_spans, list):
            frame_spans = []
        if not isinstance(long_clusters, list):
            long_clusters = []

        keep_idx = list(range(init_len))
        valid_frames = sorted(set(int(fid) for fid in selected_frames if 0 <= int(fid) < total_frames))
        for fid in valid_frames:
            if fid >= len(frame_spans):
                continue
            span = frame_spans[fid]
            keep_idx.extend(range(int(span["visual_start"]), int(span["visual_end"])))
            keep_idx.extend(range(int(span["caption_start"]), int(span["caption_end"])))
        keep_idx = sorted(set(keep_idx))

        valid_long_clusters = sorted(
            set(
                int(cid)
                for cid in (selected_long_cluster_ids or [])
                if 0 <= int(cid) < len(long_clusters)
            )
        )

        selected_lower: Dict[int, Dict[str, torch.Tensor]] = {}
        for lid, entry in lower_kv.items():
            k = entry["k"]
            v = entry["v"]
            seq_len = int(k.shape[-2])
            question_len = max(0, min(int(append_question_tokens), seq_len))
            question_start = seq_len - question_len
            safe = [x for x in keep_idx if 0 <= x < question_start]
            if question_len > 0:
                safe.extend(list(range(question_start, seq_len)))
            if not safe:
                safe = list(range(min(seq_len, init_len)))
            idx = torch.tensor(safe, device=k.device, dtype=torch.long)
            k_sel = k.index_select(dim=-2, index=idx)
            v_sel = v.index_select(dim=-2, index=idx)

            # Insert long-term cluster centroids after init prompt tokens, and
            # before short-term frame/caption tokens and trailing question tokens.
            if len(valid_long_clusters) > 0:
                long_k_parts: List[torch.Tensor] = []
                long_v_parts: List[torch.Tensor] = []
                for cid in valid_long_clusters:
                    cluster = long_clusters[cid]
                    c_lkv = cluster.get("lower_kv")
                    if not isinstance(c_lkv, dict):
                        continue
                    c_entry = c_lkv.get(int(lid))
                    if not isinstance(c_entry, dict):
                        continue
                    ck = c_entry.get("k")
                    cv = c_entry.get("v")
                    if not isinstance(ck, torch.Tensor) or not isinstance(cv, torch.Tensor):
                        continue
                    long_k_parts.append(ck.to(device=k.device, dtype=k.dtype))
                    long_v_parts.append(cv.to(device=v.device, dtype=v.dtype))

                if long_k_parts:
                    k_long = torch.cat(long_k_parts, dim=-2)
                    v_long = torch.cat(long_v_parts, dim=-2)
                    if question_len > 0 and int(k_sel.shape[-2]) >= question_len:
                        split = int(k_sel.shape[-2]) - question_len
                        k_ctx = k_sel[:, :, :split, :]
                        v_ctx = v_sel[:, :, :split, :]
                        k_q = k_sel[:, :, split:, :]
                        v_q = v_sel[:, :, split:, :]
                    else:
                        k_ctx = k_sel
                        v_ctx = v_sel
                        k_q = None
                        v_q = None

                    init_prefix_len = min(max(0, int(init_len)), int(k_ctx.shape[-2]))
                    k_init = k_ctx[:, :, :init_prefix_len, :]
                    v_init = v_ctx[:, :, :init_prefix_len, :]
                    k_short = k_ctx[:, :, init_prefix_len:, :]
                    v_short = v_ctx[:, :, init_prefix_len:, :]

                    k_parts = [k_init, k_long, k_short]
                    v_parts = [v_init, v_long, v_short]
                    if isinstance(k_q, torch.Tensor) and isinstance(v_q, torch.Tensor):
                        k_parts.append(k_q)
                        v_parts.append(v_q)
                    k_sel = torch.cat(k_parts, dim=-2)
                    v_sel = torch.cat(v_parts, dim=-2)
            selected_lower[lid] = {"k": k_sel, "v": v_sel}

        selected_caption_tokens = 0
        for fid in valid_frames:
            if fid < len(frame_spans):
                span = frame_spans[fid]
                selected_caption_tokens += max(0, int(span["caption_end"]) - int(span["caption_start"]))
        long_visual_tokens = len(valid_long_clusters) * int(self.config["n_frame_tokens"])
        self._dbg(
            f"selected_cache_rebuild: frames={valid_frames} "
            f"visual_tokens={len(valid_frames) * int(self.config['n_frame_tokens'])} "
            f"long_clusters={valid_long_clusters} long_visual_tokens={long_visual_tokens} "
            f"caption_tokens={selected_caption_tokens} seq_tokens={len(keep_idx)}"
        )
        self._dbg_mem("selected_cache_rebuild:end", device)
        return selected_lower, {}

    def _build_answer_decode_lower_cache(
        self,
        selected_frames: List[int],
        selected_long_cluster_ids: Optional[List[int]],
        lower_with_prompt: Dict[int, Dict[str, torch.Tensor]],
        prompt_len: int,
        device: str,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Select evidence while retaining the already-prefilled question tail."""
        if not isinstance(lower_with_prompt, dict) or not lower_with_prompt:
            raise RuntimeError("OneVision answer decode requires a non-empty shallow KV cache")
        if bool(self.config.get("full_kv_mode", False)):
            frame_spans = self.state.get("frame_spans")
            expected_frames = list(range(len(frame_spans))) if isinstance(frame_spans, list) else []
            if list(selected_frames) != expected_frames or selected_long_cluster_ids:
                raise RuntimeError(
                    "FullKV OneVision must decode from every historical frame without clusters"
                )
            return lower_with_prompt
        lower_sel, _ = self._build_selected_split_cache(
            selected_frames=selected_frames,
            device=device,
            source_lower_kv=lower_with_prompt,
            append_question_tokens=prompt_len,
            selected_long_cluster_ids=selected_long_cluster_ids,
        )
        if self.config.get("debug"):
            init_len = int(self.state.get("init_len", 0))
            tpf = int(self.config["n_frame_tokens"])
            frame_hidden = self.state.get("frame_hidden_l8")
            frame_spans = self.state.get("frame_spans")
            long_clusters = self.state.get("long_clusters")
            total_frames = 0 if frame_hidden is None else int(frame_hidden.shape[0])
            valid_frames = sorted(set(int(fid) for fid in selected_frames if 0 <= int(fid) < total_frames))
            valid_long_clusters = sorted(
                set(
                    int(cid)
                    for cid in (selected_long_cluster_ids or [])
                    if isinstance(long_clusters, list) and 0 <= int(cid) < len(long_clusters)
                )
            )
            caption_tokens = 0
            if isinstance(frame_spans, list):
                for fid in valid_frames:
                    if fid < len(frame_spans):
                        span = frame_spans[fid]
                        caption_tokens += max(0, int(span["caption_end"]) - int(span["caption_start"]))
            long_visual_tokens = len(valid_long_clusters) * tpf
            expected = (
                init_len
                + len(valid_frames) * tpf
                + long_visual_tokens
                + caption_tokens
                + max(0, int(prompt_len))
            )
            layer_lens = sorted({int(entry["k"].shape[-2]) for entry in lower_sel.values()})
            self._dbg(
                "answer_decode_lower_cache: "
                f"frames={valid_frames} long_clusters={valid_long_clusters} "
                f"prompt_len={prompt_len} caption_tokens={caption_tokens} "
                f"expected_tokens={expected} layer_lens={layer_lens}"
            )
        return lower_sel

    def _decode_from_reused_shallow_prefill(
        self,
        prompt_ids: torch.Tensor,
        hidden_after_prune: torch.Tensor,
        selected_frames: List[int],
        selected_long_cluster_ids: Optional[List[int]],
        selected_lower_with_prompt_kv: Dict[int, Dict[str, torch.Tensor]],
        device: str,
    ) -> Tuple[str, float, float, int]:
        layers, embed_tokens, norm, lm_head = self._get_lm_components()
        if lm_head is None:
            raise RuntimeError("lm_head is None")
        total_layers = len(layers)
        prune = min(self.config["prune_layer"], total_layers)

        with torch.inference_mode():
            t0 = time.perf_counter()
            dtype = hidden_after_prune.dtype
            if bool(self.config.get("full_kv_mode", False)):
                # FullKV has already evaluated the complete context through all
                # decoder layers. The token-wise norm/head only needs the final
                # question position to produce the first generated token.
                upper_input = hidden_after_prune[:, -1:, :].to(device=device, dtype=dtype)
            else:
                ctx_parts: List[torch.Tensor] = []
                init_hidden_l8 = self.state.get("init_hidden_l8")
                if isinstance(init_hidden_l8, torch.Tensor) and init_hidden_l8.numel() > 0:
                    ctx_parts.append(init_hidden_l8.to(device=device, dtype=dtype))

                long_clusters = self.state.get("long_clusters")
                if isinstance(long_clusters, list) and len(long_clusters) > 0:
                    valid_long_clusters = sorted(
                        set(
                            int(cid)
                            for cid in (selected_long_cluster_ids or [])
                            if 0 <= int(cid) < len(long_clusters)
                        )
                    )
                    long_parts: List[torch.Tensor] = []
                    for cid in valid_long_clusters:
                        cluster = long_clusters[cid]
                        h = cluster.get("hidden_l8")
                        if isinstance(h, torch.Tensor) and h.numel() > 0:
                            long_parts.append(h.reshape(-1, h.shape[-1]).to(device=device, dtype=dtype))
                    if long_parts:
                        ctx_parts.append(torch.cat(long_parts, dim=0))

                frame_hidden_l8 = self.state.get("frame_hidden_l8")
                if isinstance(frame_hidden_l8, torch.Tensor) and frame_hidden_l8.numel() > 0:
                    total_frames = int(frame_hidden_l8.shape[0])
                    valid_frames = sorted(set(int(fid) for fid in selected_frames if 0 <= int(fid) < total_frames))
                    if len(valid_frames) > 0:
                        caption_hidden_l8 = self.state.get("frame_caption_hidden_l8")
                        per_frame_parts: List[torch.Tensor] = []
                        for fid in valid_frames:
                            visual_hidden = frame_hidden_l8[fid].reshape(-1, frame_hidden_l8.shape[-1])
                            per_frame_parts.append(visual_hidden.to(device=device, dtype=dtype))
                            if isinstance(caption_hidden_l8, list) and fid < len(caption_hidden_l8):
                                cap_hidden = caption_hidden_l8[fid]
                                if isinstance(cap_hidden, torch.Tensor) and cap_hidden.numel() > 0:
                                    per_frame_parts.append(cap_hidden.to(device=device, dtype=dtype))
                        if per_frame_parts:
                            ctx_parts.append(torch.cat(per_frame_parts, dim=0))

                ctx_parts.append(hidden_after_prune[0].to(device=device, dtype=dtype))
                upper_input = torch.cat(ctx_parts, dim=0).unsqueeze(0)
            hidden, upper_cache = self._forward_layer_range(
                upper_input,
                prune,
                total_layers,
                {},
                causal=True,
            )
            if norm is not None:
                hidden = norm(hidden)
            logits = lm_head(hidden[:, -1:, :])[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if bool(self.config.get("latency_sync_cuda")) and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(device))
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            generated = [next_token]
            eos_id = self.tokenizer.eos_token_id
            eos_set = set(eos_id) if isinstance(eos_id, list) else {eos_id}
            lower_cache = selected_lower_with_prompt_kv
            for _ in range(max(self.config["max_new_tokens"] - 1, 0)):
                if (
                    not bool(self.config.get("force_exact_new_tokens"))
                    and int(next_token.item()) in eos_set
                ):
                    break
                token_embed = embed_tokens(next_token).to(dtype=torch.float16, device=device)
                hidden, lower_cache = self._forward_lower_layers_raw(
                    hidden_states=token_embed,
                    start_layer=0,
                    end_layer=prune,
                    past_raw_kv=lower_cache,
                    update_cache=True,
                )
                hidden, upper_cache = self._forward_layer_range(hidden, prune, total_layers, upper_cache)
                if norm is not None:
                    hidden = norm(hidden)
                logits = lm_head(hidden[:, -1:, :])[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated.append(next_token)

            out_ids = torch.cat([prompt_ids, *generated], dim=1)
            text = self._decode_generated(out_ids, prompt_ids.shape[1])
            decode_total_ms = (time.perf_counter() - t0) * 1000.0
            generated_tokens = len(generated)
            return text, ttft_ms, decode_total_ms, generated_tokens
