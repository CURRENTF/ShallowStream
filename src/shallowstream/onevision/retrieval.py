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


class OneVisionRetrievalMixin:
    def _tokenize(self, text: str, device: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(device)

    def _extract_core_question_text(self, prompt_text: str) -> str:
        text = str(prompt_text or "")
        m = re.search(r"Question:\s*(.*?)(?:\n\s*Options:|\n\s*The best option is:|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            core = m.group(1).strip()
            if core:
                return core
        return text.strip()

    def _latest_unit_score_observation_layer_indices(
        self,
        prune: int,
    ) -> List[int]:
        configured = self.config.get("observe_latest_unit_score_layers", [])
        if configured in (None, []):
            return []
        if not isinstance(configured, list):
            raise ValueError("observe_latest_unit_score_layers must be a list")
        layer_numbers = sorted({int(value) for value in configured})
        invalid = [number for number in layer_numbers if not 1 <= number <= int(prune)]
        if invalid:
            raise ValueError(
                "observe_latest_unit_score_layers must contain one-based layers "
                f"within [1, {int(prune)}], got {invalid}"
            )
        return [number - 1 for number in layer_numbers]

    def _forward_question_once_for_retrieval_and_prefill(
        self,
        question_text: str,
        generation_prompt: str,
        device: str,
        collect_all_layers: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[int, Dict[str, torch.Tensor]], torch.Tensor]:
        if not str(question_text).strip():
            raise ValueError("OneVision retrieval question must not be empty")
        prompt_ids = self._tokenize(generation_prompt, device)
        layers, embed_tokens, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        if prune <= 0:
            raise RuntimeError("prune_layer must be >= 1 for internal retrieval")

        with torch.inference_mode():
            hidden = embed_tokens(prompt_ids).to(dtype=torch.float16, device=device)
            q_layer_vecs: List[torch.Tensor] = []
            lower_kv = self.state.get("lower_kv")
            rope_theta = self._rope_base()
            q_len = int(hidden.shape[1])
            prompt_cache: Dict[int, Dict[str, torch.Tensor]] = {}
            preserve_fullkv_stream = bool(
                self.config.get("full_kv_mode", False)
                and self.config.get("fullkv_preserve_stream_history", False)
            )
            if preserve_fullkv_stream:
                archived_lower_kv: Dict[int, Dict[str, torch.Tensor]] = {}
                for layer_idx, entry in (lower_kv or {}).items():
                    archived_lower_kv[int(layer_idx)] = {
                        "k": entry["k"].detach().to(device="cpu"),
                        "v": entry["v"].detach().to(device="cpu"),
                    }
                if len(archived_lower_kv) != prune:
                    raise RuntimeError(
                        "Persistent FullKV OneVision archive does not cover every decoder layer: "
                        f"archive={len(archived_lower_kv)}, expected={prune}"
                    )
                self.state["lower_kv"] = archived_lower_kv
                lower_kv = archived_lower_kv
            consume_lower_kv = bool(
                self.config.get("full_kv_mode", False) and not preserve_fullkv_stream
            )
            observed_layer_indices = set(
                self._latest_unit_score_observation_layer_indices(prune)
            )
            capture_history_decay = self._task_gate_mode() == "history_layer_decay"
            history_decay_layers: Dict[int, Dict[str, torch.Tensor]] = {}
            query_q_layers: Dict[int, Dict[str, torch.Tensor]] = {}

            for idx in range(prune):
                layer = layers[idx]
                attn = layer.self_attn

                residual = hidden
                h_norm = layer.input_layernorm(hidden)
                bsz, q_len, _ = h_norm.shape
                num_heads, num_kv_heads, head_dim = self._attention_shape(attn)

                q_lin = attn.q_proj(h_norm).view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
                k_lin = attn.k_proj(h_norm).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
                v_lin = attn.v_proj(h_norm).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

                # Compute question vector only when needed:
                # - always for prune-1 (retrieval scoring path)
                # - for all layers only when debug_similarity is enabled
                if (
                    collect_all_layers
                    or idx == (prune - 1)
                    or idx in observed_layer_indices
                ):
                    q_flat = q_lin[0].transpose(0, 1).reshape(q_len, -1)
                    q_vec = q_flat.mean(dim=0)
                    q_layer_vecs.append(torch.nn.functional.normalize(q_vec, dim=-1))
                else:
                    q_layer_vecs.append(torch.empty(0, device=device, dtype=h_norm.dtype))

                past_entry = lower_kv.get(idx) if isinstance(lower_kv, dict) else None
                if past_entry is not None:
                    past_k = past_entry["k"]
                    past_v = past_entry["v"]
                    if past_k.device != h_norm.device or past_k.dtype != h_norm.dtype:
                        past_k = past_k.to(device=h_norm.device, dtype=h_norm.dtype)
                    if past_v.device != h_norm.device or past_v.dtype != h_norm.dtype:
                        past_v = past_v.to(device=h_norm.device, dtype=h_norm.dtype)
                else:
                    past_k = torch.empty((bsz, num_kv_heads, 0, head_dim), device=h_norm.device, dtype=h_norm.dtype)
                    past_v = torch.empty((bsz, num_kv_heads, 0, head_dim), device=h_norm.device, dtype=h_norm.dtype)

                past_len = int(past_k.shape[-2])
                past_pos_value = past_entry.get("pos") if isinstance(past_entry, dict) else None
                if isinstance(past_pos_value, torch.Tensor):
                    past_pos = past_pos_value.to(
                        device=h_norm.device,
                        dtype=torch.long,
                    ).reshape(-1)
                    if int(past_pos.numel()) != past_len:
                        raise RuntimeError(
                            "Invalid OneVision retrieval-cache positions: "
                            f"got {int(past_pos.numel())}, expected {past_len}"
                        )
                else:
                    # Rebuilt evidence caches intentionally use compact,
                    # contiguous positions and omit the absolute-position field.
                    past_pos = torch.arange(
                        past_len,
                        device=h_norm.device,
                        dtype=torch.long,
                    )
                cur_start = int(past_pos[-1].item()) + 1 if past_len > 0 else 0
                cur_pos = torch.arange(
                    cur_start,
                    cur_start + q_len,
                    device=h_norm.device,
                    dtype=torch.long,
                )
                if self._retrieval_score_strategy() == "shallow_layer_token_vote":
                    query_q_layers[idx] = {
                        "q": q_lin.detach(),
                        "positions": cur_pos.detach(),
                    }
                if capture_history_decay:
                    history_decay_layers[idx] = {"q": q_lin.detach(), "positions": cur_pos.detach()}
                total_k = torch.cat([past_k, k_lin], dim=-2)
                total_v = torch.cat([past_v, v_lin], dim=-2)
                total_pos = torch.cat([past_pos, cur_pos], dim=0)

                if consume_lower_kv and torch.cuda.is_available():
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(h_norm.device)
                    if free_bytes < 256 * 1024 * 1024:
                        torch.cuda.empty_cache()

                attn_output = self._lower_attend_with_rekv_sink(
                    q=q_lin,
                    total_k=total_k,
                    total_v=total_v,
                    cur_pos=cur_pos,
                    total_pos=total_pos,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                    n_local_tokens=self._local_tail_tokens(q_len),
                    sink_len=int(self.state.get("init_len", 0)),
                )
                attn_output = attn.o_proj(attn_output)

                hidden = residual + attn_output
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden

                prompt_cache[idx] = {
                    "k": total_k.detach(),
                    "v": total_v.detach(),
                    "pos": total_pos.detach(),
                }
                if consume_lower_kv and isinstance(lower_kv, dict):
                    lower_kv.pop(idx, None)

        self.state["last_history_decay_query"] = ({
            "layers": history_decay_layers,
            "prompt_ids": prompt_ids.detach(),
            "generation_prompt": str(generation_prompt),
        } if capture_history_decay else {})
        self.state["last_retrieval_query_q"] = query_q_layers
        return q_layer_vecs, hidden.detach(), prompt_cache, prompt_ids

    def _compute_question_query_vectors_per_layer(self, question_text: str, device: str) -> List[torch.Tensor]:
        generation_prompt = question_text + str(self.config.get("assistant_suffix", ""))
        q_layer_vecs, _, _, _ = self._forward_question_once_for_retrieval_and_prefill(
            question_text=question_text,
            generation_prompt=generation_prompt,
            device=device,
            collect_all_layers=bool(self.config.get("debug_similarity")),
        )
        return q_layer_vecs

    def _compute_layer_frame_key_vectors(self, layer_idx: int, device: str) -> torch.Tensor:
        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if lower_kv is None:
            return torch.empty((0, 0), device=device, dtype=torch.float16)
        if not isinstance(frame_spans, list) or len(frame_spans) == 0:
            return torch.empty((0, 0), device=device, dtype=torch.float16)

        layers, _, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        if prune <= 0 or layer_idx < 0 or layer_idx >= prune:
            return torch.empty((0, 0), device=device, dtype=torch.float16)

        entry = lower_kv.get(layer_idx)
        if entry is None:
            return torch.empty((0, 0), device=device, dtype=torch.float16)

        k = entry["k"].to(device=device)
        tpf = int(self.config["n_frame_tokens"])
        seq_len = int(k.shape[-2])
        usable_spans = frame_spans
        if len(usable_spans) <= 0:
            return torch.empty((0, 0), device=device, dtype=torch.float16)

        attn = layers[layer_idx].self_attn
        num_heads, num_kv_heads, _head_dim = self._attention_shape(attn)
        frame_vecs: List[torch.Tensor] = []
        for span in usable_spans:
            st = int(span.get("visual_start", -1))
            ed = int(span.get("visual_end", -1))
            if st < 0 or ed <= st or ed > seq_len:
                return torch.empty((0, 0), device=device, dtype=torch.float16)
            k_frame = k[:, :, st:ed, :]
            if int(k_frame.shape[-2]) != tpf:
                return torch.empty((0, 0), device=device, dtype=torch.float16)
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k_frame = self._repeat_kv(k_frame, n_rep)
            frame_vec = k_frame[0].transpose(0, 1).reshape(tpf, -1).mean(dim=0)
            frame_vecs.append(torch.nn.functional.normalize(frame_vec, dim=-1))
        if not frame_vecs:
            return torch.empty((0, 0), device=device, dtype=torch.float16)
        return torch.stack(frame_vecs, dim=0)

    def _build_latest_unit_score_observation(
        self,
        *,
        q_layer_vecs: List[torch.Tensor],
        prune_frame_vecs: torch.Tensor,
        device: str,
    ) -> Dict[str, Any]:
        layer_indices = self._latest_unit_score_observation_layer_indices(
            len(q_layer_vecs)
        )
        if not layer_indices:
            return {}
        if prune_frame_vecs.numel() == 0:
            raise RuntimeError(
                "latest-unit layer observation requires sampled-frame key vectors"
            )

        latest_frame_id = int(prune_frame_vecs.shape[0]) - 1
        source_ids = self.state.get("frame_source_ids")
        latest_timestamp = (
            float(source_ids[latest_frame_id])
            if isinstance(source_ids, list) and latest_frame_id < len(source_ids)
            else float(latest_frame_id)
        )
        layer_scores: Dict[str, Any] = {}
        for layer_idx in layer_indices:
            frame_vecs = (
                prune_frame_vecs
                if layer_idx == len(q_layer_vecs) - 1
                else self._compute_layer_frame_key_vectors(layer_idx, device)
            )
            query_vec = q_layer_vecs[layer_idx]
            if frame_vecs.numel() == 0 or query_vec.numel() == 0:
                raise RuntimeError(
                    "latest-unit layer observation is missing Q/K vectors for "
                    f"layer {layer_idx + 1}"
                )
            score = float(
                torch.dot(
                    frame_vecs[-1].float(),
                    query_vec.to(device=frame_vecs.device).float(),
                ).item()
            )
            if not math.isfinite(score):
                raise RuntimeError(
                    "latest-unit layer observation produced a non-finite score"
                )
            layer_scores[str(layer_idx + 1)] = {
                "layer_number": int(layer_idx + 1),
                "layer_index": int(layer_idx),
                "score": score,
            }
        return {
            "metric": "shallow_qk_cosine",
            "representation": (
                "normalized_latest_sampled_frame_k_dot_normalized_question_q"
            ),
            "query_source": "retrieval_query_vector",
            "unit_granularity": "onevision_sampled_frame",
            "latest_unit_frame_id": latest_frame_id,
            "latest_unit_sample_index": latest_frame_id,
            "latest_unit_timestamp": latest_timestamp,
            "layers": layer_scores,
        }

    def _compute_layer_subtitle_key_vectors(self, layer_idx: int, device: str) -> Tuple[torch.Tensor, List[bool]]:
        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if lower_kv is None:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []
        if not isinstance(frame_spans, list) or len(frame_spans) == 0:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []

        layers, _, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        if prune <= 0 or layer_idx < 0 or layer_idx >= prune:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []

        entry = lower_kv.get(layer_idx)
        if entry is None:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []

        k = entry["k"].to(device=device)
        seq_len = int(k.shape[-2])
        usable_spans = frame_spans
        if len(usable_spans) <= 0:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []

        attn = layers[layer_idx].self_attn
        num_heads, num_kv_heads, head_dim = self._attention_shape(attn)
        vec_dim = int(num_heads * head_dim)

        subtitle_vecs: List[torch.Tensor] = []
        caption_mask: List[bool] = []
        for span in usable_spans:
            st = int(span.get("caption_start", -1))
            ed = int(span.get("caption_end", -1))
            if st < 0 or ed <= st or st >= seq_len:
                subtitle_vecs.append(torch.zeros((vec_dim,), dtype=k.dtype, device=device))
                caption_mask.append(False)
                continue
            ed = min(ed, seq_len)
            if ed <= st:
                subtitle_vecs.append(torch.zeros((vec_dim,), dtype=k.dtype, device=device))
                caption_mask.append(False)
                continue
            k_cap = k[:, :, st:ed, :]
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k_cap = self._repeat_kv(k_cap, n_rep)
            cap_vec = k_cap[0].transpose(0, 1).reshape(int(k_cap.shape[-2]), -1).mean(dim=0)
            subtitle_vecs.append(torch.nn.functional.normalize(cap_vec, dim=-1))
            caption_mask.append(True)

        if not subtitle_vecs:
            return torch.empty((0, 0), device=device, dtype=torch.float16), []
        return torch.stack(subtitle_vecs, dim=0), caption_mask

    def _select_frames_by_question(
        self,
        question_text: str,
        device: str,
        q_layer_vecs: Optional[List[torch.Tensor]] = None,
        video_path: Optional[str] = None,
        task_gate_text: Optional[str] = None,
    ) -> List[int]:
        self.state["last_latest_unit_score_observation"] = {}
        self.state["last_history_decay_observation"] = {}
        if bool(self.config.get("full_kv_mode", False)):
            frame_spans = self.state.get("frame_spans")
            selected = list(range(len(frame_spans))) if isinstance(frame_spans, list) else []
            self.state["last_selection"] = {
                "policy": "full_kv",
                "keep_idx": selected,
                "long_cluster_indices": [],
            }
            return selected
        if self._evidence_retrieval_backend() == "siglip":
            return self._apply_task_gate_to_frames(
                [],
                question_text,
                question_text=("" if task_gate_text is None else str(task_gate_text)),
            )
        if q_layer_vecs is None:
            q_layer_vecs = self._compute_question_query_vectors_per_layer(question_text, device)
        if len(q_layer_vecs) == 0:
            return []
        prune = len(q_layer_vecs)
        if self._task_gate_mode() == "history_layer_decay":
            gate_question = str(task_gate_text or "").strip()
            if not gate_question:
                gate_question = self._extract_core_question_text(question_text)
            self.state["last_history_decay_observation"] = self._build_history_decay_observation(gate_question)
        frame_vec = self._compute_layer_frame_key_vectors(prune - 1, device)
        if frame_vec.numel() == 0:
            return []
        v_w, s_w = self._get_retrieval_weights()
        subtitle_mask = self._get_frame_caption_available(int(frame_vec.shape[0]))
        q_vec = q_layer_vecs[prune - 1].to(dtype=frame_vec.dtype)
        temp = float(self.config.get("retrieval_temperature", 1.0))
        if temp <= 0:
            temp = 1.0
        visual_scores = torch.matmul(frame_vec, q_vec) / temp
        self.state["last_latest_unit_score_observation"] = (
            self._build_latest_unit_score_observation(
                q_layer_vecs=q_layer_vecs,
                prune_frame_vecs=frame_vec,
                device=device,
            )
        )
        subtitle_scores: Optional[torch.Tensor] = None
        if s_w > 0.0:
            subtitle_vec, subtitle_mask_layer = self._compute_layer_subtitle_key_vectors(prune - 1, device)
            if subtitle_vec.numel() > 0 and subtitle_vec.shape == frame_vec.shape:
                subtitle_scores = torch.matmul(subtitle_vec.to(dtype=frame_vec.dtype), q_vec) / temp
                subtitle_mask = subtitle_mask_layer
        scores = self._fuse_retrieval_scores(
            visual_scores=visual_scores,
            subtitle_scores=subtitle_scores,
            caption_mask=subtitle_mask,
        )

        total_frames = int(scores.numel())
        search_last_n_cfg = max(0, int(self.config.get("retrieval_search_last_n_frames", 0)))
        candidate_start = self._short_window_start(total_frames)
        candidate_idx = list(range(candidate_start, total_frames))
        if len(candidate_idx) == 0:
            return []
        candidate_n = len(candidate_idx)
        candidate_min = candidate_idx[0]
        candidate_max = candidate_idx[-1]
        candidate_idx_t = torch.tensor(candidate_idx, device=scores.device, dtype=torch.long)

        topk_cfg = max(0, int(self.config.get("retrieval_topk_frames", 0)))
        topk = min(topk_cfg, candidate_n)
        recent_cfg = max(0, int(self.config.get("retrieval_recent_frames", 0)))
        recent_n = min(recent_cfg, candidate_n)
        if topk <= 0 and recent_n <= 0:
            return []
        score_order = str(self.config.get("retrieval_score_order", "lowest")).lower()
        if score_order not in ("highest", "lowest"):
            raise ValueError(f"Unsupported retrieval_score_order={score_order!r}; expected 'highest' or 'lowest'")

        self._observe_retrieval(
            scores=scores,
            candidate_idx_t=candidate_idx_t,
            recent_n=recent_n,
            score_order=score_order,
            video_path=video_path,
        )

        # Stage-1: short-term frames (always keep latest N)
        recent_idx: List[int] = []
        if recent_n > 0:
            recent_idx = candidate_idx[-recent_n:]

        # Stage-2: important frames from non-recent region
        important_seed_idx: List[int] = []
        vote_stats: Dict[str, Any] = {}
        if topk > 0 and self._retrieval_score_strategy() == "shallow_layer_token_vote":
            historical_candidates = (
                candidate_idx[: candidate_n - recent_n]
                if recent_n > 0 and candidate_n > recent_n
                else candidate_idx
            )
            important_seed_idx, vote_stats = (
                self._select_shallow_layer_token_vote_frames(
                    candidate_frames=historical_candidates,
                    frame_vecs=frame_vec,
                    frame_budget=topk,
                    device=device,
                )
            )
        elif topk > 0:
            if recent_n > 0 and candidate_n > recent_n:
                important_candidate_idx = candidate_idx_t[: candidate_n - recent_n]
                important_candidate_scores = scores.index_select(0, important_candidate_idx)
                k_imp = min(topk, important_candidate_scores.numel())
                if k_imp > 0:
                    local_top = torch.topk(
                        important_candidate_scores,
                        k=k_imp,
                        largest=(score_order == "highest"),
                    ).indices
                    important_seed_idx = important_candidate_idx.index_select(0, local_top).detach().cpu().tolist()
            else:
                candidate_scores = scores.index_select(0, candidate_idx_t)
                k_imp = min(topk, candidate_scores.numel())
                if k_imp > 0:
                    local_top = torch.topk(
                        candidate_scores,
                        k=k_imp,
                        largest=(score_order == "highest"),
                    ).indices
                    important_seed_idx = candidate_idx_t.index_select(0, local_top).detach().cpu().tolist()

        # Expand each important frame with configurable temporal neighbors.
        expand_prev = max(0, int(self.config.get("retrieval_expand_prev_frames", 1)))
        expand_next = max(0, int(self.config.get("retrieval_expand_next_frames", 0)))
        stride_prev = max(1, int(self.config.get("retrieval_expand_prev_stride", 1)))
        stride_next = max(1, int(self.config.get("retrieval_expand_next_stride", 1)))
        important_idx = expand_temporal_neighbors(
            important_seed_idx,
            candidate_idx,
            previous=expand_prev,
            following=expand_next,
            previous_stride=stride_prev,
            following_stride=stride_next,
        )

        keep_idx = sorted(set(recent_idx).union(important_idx))
        if len(keep_idx) == 0:
            return []
        src_ids = self.state.get("frame_source_ids", [])
        selected_src_ids = [src_ids[i] if isinstance(src_ids, list) and 0 <= i < len(src_ids) else -1.0 for i in keep_idx]
        selected_scores = scores[torch.tensor(keep_idx, device=scores.device, dtype=torch.long)].detach().float().cpu().tolist()
        self._dbg(
            f"select_frames(prune_qk): total={scores.numel()} keep={keep_idx} "
            f"temp={temp:.3f} topk={topk} order={score_order} "
            f"recent={recent_n} retrieval_topk_frames={topk_cfg} "
            f"candidate_last_n={search_last_n_cfg} candidate_range=[{candidate_min},{candidate_max}] "
            f"expand_prev={expand_prev}(stride={stride_prev}) "
            f"expand_next={expand_next}(stride={stride_next}) "
            f"weights=(v={v_w:.2f},s={s_w:.2f}) "
            f"prune_layer={int(self.config.get('prune_layer', 8))}"
        )
        selected_src_ids_fmt = [round(float(x), 3) for x in selected_src_ids]
        self._dbg_frames(
            f"selected idx={keep_idx} source_ts_s={selected_src_ids_fmt} "
            f"recent_idx={recent_idx} important_seed_idx={sorted(int(x) for x in important_seed_idx)} "
            f"important_expanded_idx={important_idx} "
            f"candidate_ts_range=[{(round(float(src_ids[0]), 3) if isinstance(src_ids, list) and len(src_ids) > 0 else 'NA')},"
            f"{(round(float(src_ids[-1]), 3) if isinstance(src_ids, list) and len(src_ids) > 0 else 'NA')}] "
            f"order={score_order} scores={[round(float(x), 4) for x in selected_scores]}"
        )
        selected_long_cluster_idx, selected_long_cluster_scores = self._select_long_clusters_by_query(
            layer_idx=prune - 1,
            q_vec=q_vec,
            temp=temp,
            score_order=score_order,
            device=device,
        )
        self.state["last_selection"] = {
            "keep_idx": [int(x) for x in keep_idx],
            "recent_idx": [int(x) for x in recent_idx],
            "important_seed_idx": [int(x) for x in important_seed_idx],
            "important_expanded_idx": [int(x) for x in important_idx],
            "long_cluster_indices": [int(x) for x in selected_long_cluster_idx],
            "long_cluster_scores": [float(x) for x in selected_long_cluster_scores],
            "expand_prev_frames": int(expand_prev),
            "expand_next_frames": int(expand_next),
            "expand_prev_stride": int(stride_prev),
            "expand_next_stride": int(stride_next),
            "selected_scores": [float(x) for x in selected_scores],
            "selected_visual_scores": [float(x) for x in visual_scores[torch.tensor(keep_idx, device=visual_scores.device, dtype=torch.long)].detach().float().cpu().tolist()],
            "selected_subtitle_scores": (
                [float(x) for x in subtitle_scores[torch.tensor(keep_idx, device=subtitle_scores.device, dtype=torch.long)].detach().float().cpu().tolist()]
                if isinstance(subtitle_scores, torch.Tensor)
                else []
            ),
            "selected_caption_available": [bool(subtitle_mask[i]) if i < len(subtitle_mask) else False for i in keep_idx],
            "selected_source_ts_s": [float(x) for x in selected_src_ids],
            "score_order": score_order,
            "candidate_last_n_frames": int(search_last_n_cfg),
            "candidate_range_idx": [int(candidate_min), int(candidate_max)],
            "weights": {
                "visual": float(v_w),
                "subtitle": float(s_w),
            },
            **vote_stats,
        }
        if selected_long_cluster_idx:
            self._dbg_frames(
                "selected long clusters="
                + str(
                    [
                        {
                            "cluster_idx": int(cid),
                            "score": round(float(sc), 4),
                        }
                        for cid, sc in zip(selected_long_cluster_idx, selected_long_cluster_scores)
                    ]
                )
            )
        layer_scores: Optional[List[List[float]]] = None
        if self.config.get("debug_similarity"):
            layer_scores = []
            temp_safe = temp if temp > 0 else 1.0
            for l in range(prune):
                fv = self._compute_layer_frame_key_vectors(l, device)
                if fv.numel() == 0:
                    layer_scores.append([])
                    continue
                qv = q_layer_vecs[l].to(dtype=fv.dtype)
                sv = torch.matmul(fv, qv) / temp_safe
                sc: Optional[torch.Tensor] = None
                cm: Optional[List[bool]] = None
                if s_w > 0.0:
                    cv, cm = self._compute_layer_subtitle_key_vectors(l, device)
                    cv_t = cv.to(dtype=fv.dtype) if cv.numel() > 0 and cv.shape == fv.shape else None
                    if isinstance(cv_t, torch.Tensor) and cv_t.numel() > 0 and cv_t.shape == fv.shape:
                        sc = torch.matmul(cv_t.to(dtype=fv.dtype), qv) / temp_safe
                sf = self._fuse_retrieval_scores(sv, sc, cm)
                layer_scores.append(sf.detach().float().cpu().tolist())
        self._dump_similarity_debug(
            question_text=question_text,
            scores=scores,
            source_ts=src_ids if isinstance(src_ids, list) else [],
            keep_idx=keep_idx,
            video_path=video_path,
            recent_idx=recent_idx,
            important_seed_idx=important_seed_idx,
            important_expanded_idx=important_idx,
            layer_scores=layer_scores,
            score_order=score_order,
        )
        return self._apply_task_gate_to_frames(
            keep_idx,
            question_text,
            question_text=(
                "" if task_gate_text is None else str(task_gate_text)
            ),
            latest_unit_query_vec=q_vec,
            latest_unit_key_vec=frame_vec[-1],
            latest_unit_frame_id=total_frames - 1,
        )
