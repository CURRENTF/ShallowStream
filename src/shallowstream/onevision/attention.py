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


def _lower_cache_positions(
    entry: Optional[Dict[str, torch.Tensor]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return cache positions, accepting legacy/rebuilt contiguous caches."""
    if seq_len <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    pos = entry.get("pos") if isinstance(entry, dict) else None
    if not isinstance(pos, torch.Tensor):
        return torch.arange(seq_len, device=device, dtype=torch.long)
    pos = pos.to(device=device, dtype=torch.long).reshape(-1)
    if int(pos.numel()) != int(seq_len):
        raise RuntimeError(
            "Invalid OneVision lower-cache positions: "
            f"got {int(pos.numel())}, expected {int(seq_len)}"
        )
    return pos


class OneVisionAttentionMixin:
    def _build_causal_mask(self, tgt_len: int, past_len: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((tgt_len, past_len + tgt_len), float("-inf"), device=device, dtype=dtype)
        if past_len > 0:
            mask[:, :past_len] = 0
        if tgt_len > 0:
            causal = torch.zeros((tgt_len, tgt_len), device=device, dtype=dtype)
            upper = torch.triu(torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool), diagonal=1)
            causal = causal.masked_fill(upper, float("-inf"))
            mask[:, past_len:] = causal
        return mask.unsqueeze(0).unsqueeze(0)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        bsz, n_kv, slen, hd = x.shape
        x = x[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, hd)
        return x.reshape(bsz, n_kv * n_rep, slen, hd)

    def _rope_base(self) -> float:
        if hasattr(self.model, "config") and hasattr(self.model.config, "rope_theta"):
            return float(self.model.config.rope_theta)
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "config") and hasattr(self.model.language_model.config, "rope_theta"):
            return float(self.model.language_model.config.rope_theta)
        return 10000.0

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
        # x: (B, H, L, D), pos: (L,)
        dim = x.shape[-1]
        half = dim // 2
        if half == 0:
            return x
        inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half)))
        freqs = torch.outer(pos.to(torch.float32), inv_freq)  # (L, D/2)
        cos = torch.cos(freqs).to(dtype=x.dtype)
        sin = torch.sin(freqs).to(dtype=x.dtype)
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        x_rot1 = x1 * cos[None, None, :, :] - x2 * sin[None, None, :, :]
        x_rot2 = x1 * sin[None, None, :, :] + x2 * cos[None, None, :, :]
        if dim % 2 == 0:
            return torch.cat([x_rot1, x_rot2], dim=-1)
        return torch.cat([x_rot1, x_rot2, x[..., 2 * half :]], dim=-1)

    def _select_sink_local_kv(
        self,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_pos: torch.Tensor,
        *,
        local_tail_len: int,
        sink_len: int,
        query_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        total_len = int(total_pos.numel())
        sink_len = max(0, min(int(sink_len), total_len))
        local_tail_len = int(local_tail_len)
        query_len = int(query_len)
        if local_tail_len > 0 and local_tail_len < query_len:
            raise ValueError(
                "local tail must include the complete current query suffix: "
                f"local={local_tail_len}, query={query_len}"
            )
        if local_tail_len <= 0 or total_len <= sink_len + local_tail_len:
            return total_k, total_v, total_pos, False

        local_start = total_len - local_tail_len
        return (
            torch.cat(
                [total_k[:, :, :sink_len, :], total_k[:, :, local_start:, :]],
                dim=-2,
            ),
            torch.cat(
                [total_v[:, :, :sink_len, :], total_v[:, :, local_start:, :]],
                dim=-2,
            ),
            torch.cat([total_pos[:sink_len], total_pos[local_start:]], dim=0),
            True,
        )

    def _lower_attend_with_rekv_sink(
        self,
        q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        cur_pos: torch.Tensor,
        total_pos: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        n_local_tokens: int,
        sink_len: int,
    ) -> torch.Tensor:
        total_len = int(total_pos.numel())
        q_len = int(q.shape[2])
        sink_len_eff = max(0, min(int(sink_len), total_len))
        local_tail_len = int(n_local_tokens)

        def _flash_attend_strict(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            if q_in.device.type != "cuda":
                raise RuntimeError(f"{self.log_name} lower prefill flash-attn requires CUDA device.")
            if q_in.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"{self.log_name} lower prefill flash-attn requires fp16/bf16, got {q_in.dtype}."
                )

            flash_attn_err: Optional[Exception] = None
            try:
                from flash_attn import flash_attn_func  # type: ignore

                # flash_attn_func expects (B, S, H, D), while this code uses
                # (B, H, S, D). Keep GQA layout (Hq >= Hkv) without KV repeat.
                q_bshd = q_in.transpose(1, 2).contiguous()
                k_bshd = k_in.transpose(1, 2).contiguous()
                v_bshd = v_in.transpose(1, 2).contiguous()
                flash_kwargs: Dict[str, Any] = {
                    "dropout_p": 0.0,
                    "softmax_scale": None,
                    "causal": True,
                }
                out_bshd = flash_attn_func(q_bshd, k_bshd, v_bshd, **flash_kwargs)
                if isinstance(out_bshd, tuple):
                    out_bshd = out_bshd[0]
                return out_bshd.transpose(1, 2).contiguous()
            except Exception as exc:
                flash_attn_err = exc

            cap = torch.cuda.get_device_capability(q_in.device) if q_in.device.type == "cuda" else None
            raise RuntimeError(
                f"{self.log_name} lower prefill strict-flash failed. "
                f"flash_attn_func_error={flash_attn_err}; "
                f"device_capability={cap}; torch={torch.__version__}; "
                f"q_shape={tuple(q_in.shape)} k_shape={tuple(k_in.shape)} dtype={q_in.dtype}; "
                "causal=True"
            )

        # Keep the sink prefix and one disjoint local suffix, then run one causal
        # attention. The caller includes the current query length in the suffix
        # budget, so FlashAttention's bottom-right causal alignment is exact for
        # every token in a batched append.
        selected_k, selected_v, selected_pos, window_active = self._select_sink_local_kv(
            total_k,
            total_v,
            total_pos,
            local_tail_len=local_tail_len,
            sink_len=sink_len_eff,
            query_len=q_len,
        )
        if not window_active:
            self._bump_lower_attn_path("flash_full_causal")
        else:
            self._bump_lower_attn_path(
                "flash_sink_local_concat" if sink_len_eff > 0 else "flash_local_suffix"
            )

        q_rot = self._apply_rope(q, cur_pos, rope_theta)
        selected_k_rot = self._apply_rope(selected_k, selected_pos, rope_theta)
        attn_output = _flash_attend_strict(q_rot, selected_k_rot, selected_v)
        return attn_output.transpose(1, 2).contiguous().view(
            q.shape[0], q.shape[2], num_heads * head_dim
        )

    def _forward_lower_layers_raw(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
        past_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]],
        update_cache: bool = True,
        collect_layer_hidden: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        layers, _, _, _ = self._get_lm_components()
        new_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        rope_theta = self._rope_base()
        consume_past_cache = bool(self.config.get("full_kv_mode", False))

        for idx in range(start_layer, end_layer):
            layer = layers[idx]
            attn = layer.self_attn
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape
            num_heads, num_kv_heads, head_dim = self._attention_shape(attn)

            q = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
            k = attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
            v = attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

            past_entry = past_raw_kv.get(idx) if past_raw_kv is not None else None
            if past_entry is not None:
                past_k = past_entry["k"]
                past_v = past_entry["v"]
                if past_k.device != hidden_states.device or past_k.dtype != hidden_states.dtype:
                    past_k = past_k.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                if past_v.device != hidden_states.device or past_v.dtype != hidden_states.dtype:
                    past_v = past_v.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
            else:
                past_k = torch.empty((bsz, num_kv_heads, 0, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                past_v = torch.empty((bsz, num_kv_heads, 0, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

            past_len = int(past_k.shape[-2])
            past_pos = _lower_cache_positions(
                past_entry,
                past_len,
                hidden_states.device,
            )
            cur_start = int(past_pos[-1].item()) + 1 if past_len > 0 else 0
            cur_pos = torch.arange(
                cur_start,
                cur_start + q_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
            if hidden_states.device.type == "cuda":
                free_bytes, _total_bytes = torch.cuda.mem_get_info(hidden_states.device)
                if free_bytes < 3 * 1024**3:
                    # Growing KV concatenation temporarily needs both the old and
                    # replacement tensors; return inactive allocator blocks first.
                    torch.cuda.empty_cache()
            total_k = torch.cat([past_k, k], dim=-2)
            if hidden_states.device.type == "cuda":
                free_bytes, _total_bytes = torch.cuda.mem_get_info(hidden_states.device)
                if free_bytes < 3 * 1024**3:
                    torch.cuda.empty_cache()
            total_v = torch.cat([past_v, v], dim=-2)
            total_pos = torch.cat([past_pos, cur_pos], dim=0)

            if hidden_states.device.type == "cuda":
                free_bytes, _total_bytes = torch.cuda.mem_get_info(hidden_states.device)
                if free_bytes < 256 * 1024 * 1024:
                    # FlashAttention needs a small workspace even when the live
                    # KV tensors fit. Release only inactive allocator blocks.
                    torch.cuda.empty_cache()

            attn_output = self._lower_attend_with_rekv_sink(
                q=q,
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

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            if collect_layer_hidden is not None:
                collect_layer_hidden.append(hidden_states.detach())

            if update_cache:
                new_cache[idx] = {
                    "k": total_k.detach(),
                    "v": total_v.detach(),
                    "pos": total_pos.detach(),
                }
            else:
                new_cache[idx] = {
                    "k": past_k,
                    "v": past_v,
                    "pos": past_pos,
                }
                del total_k
                del total_v
            if consume_past_cache and isinstance(past_raw_kv, dict):
                past_raw_kv.pop(idx, None)

        return hidden_states, new_cache

    def _layer_present_to_tuple(self, present: Any, idx: int, fallback_cache: SingleLayerLegacyCache):
        if isinstance(present, SingleLayerLegacyCache):
            return present.to_legacy()
        if present is None:
            # Some implementations update cache in-place and return None.
            return fallback_cache.to_legacy()
        if isinstance(present, tuple):
            return present
        if hasattr(present, "to_legacy_cache"):
            legacy = present.to_legacy_cache()
            if isinstance(legacy, tuple) and len(legacy) > idx and isinstance(legacy[idx], tuple):
                return legacy[idx]
        raise RuntimeError(f"Unsupported present cache type: {type(present)}")

    def _get_lm_components(self):
        base = getattr(self.model, "model", None)
        language_model = getattr(base, "language_model", None) if base is not None else None
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            base = getattr(language_model, "model", language_model)
        if base is None:
            base = self.model
        layers = getattr(base, "layers", None)
        embed_tokens = getattr(base, "embed_tokens", None) or self.model.get_input_embeddings()
        norm = getattr(base, "norm", None)
        lm_head = None
        if language_model is not None:
            lm_head = getattr(language_model, "lm_head", None)
        if lm_head is None:
            lm_head = getattr(self.model, "lm_head", None)
        return layers, embed_tokens, norm, lm_head

    def _get_lm_rotary_embedding(self):
        base = getattr(self.model, "model", None)
        language_model = getattr(base, "language_model", None) if base is not None else None
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            base = getattr(language_model, "model", language_model)
        if base is None:
            base = self.model
        rotary_emb = getattr(base, "rotary_emb", None)
        if rotary_emb is None:
            raise AttributeError("Cannot resolve language-model rotary embedding")
        return rotary_emb

    def _attention_shape(self, attn: Any) -> Tuple[int, int, int]:
        config = getattr(attn, "config", None)
        if config is None:
            config = getattr(getattr(self.model, "config", None), "text_config", None)
        num_heads = getattr(attn, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            raise AttributeError("Cannot resolve attention head count")
        num_heads = int(num_heads)
        num_kv_heads = getattr(attn, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                raise AttributeError("Cannot resolve attention head dimension")
            head_dim = int(hidden_size) // num_heads
        return num_heads, int(num_kv_heads), int(head_dim)

    def _get_active_attn_implementation(self) -> str:
        return get_active_attn_implementation(self.model)

    def _uses_flash_attention_2(self) -> bool:
        return self._get_active_attn_implementation() == "flash_attention_2"

    def _forward_layer_range(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
        past_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        layers, _, _, _ = self._get_lm_components()
        rotary_emb = self._get_lm_rotary_embedding()
        new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for idx in range(start_layer, end_layer):
            layer = layers[idx]
            past_tuple = past_kv.get(idx) if past_kv is not None else None
            cache_obj = SingleLayerLegacyCache(past_tuple)
            past_len = past_tuple[0].shape[-2] if past_tuple is not None else 0
            if self._uses_flash_attention_2():
                self.state["upper_flash_layer_calls"] = int(self.state.get("upper_flash_layer_calls", 0)) + 1
                # HF FlashAttention2 applies causal masking internally for Qwen2.
                # Passing our 4D additive mask would route through an incompatible
                # code path, so keep the same causal semantics with mask=None.
                attn_mask = None
            else:
                self.state["upper_nonflash_layer_calls"] = int(self.state.get("upper_nonflash_layer_calls", 0)) + 1
                if causal:
                    attn_mask = self._build_causal_mask(
                        tgt_len=hidden_states.shape[1],
                        past_len=past_len,
                        device=str(hidden_states.device),
                        dtype=hidden_states.dtype,
                    )
                else:
                    # Full bidirectional attention on the current chunk, while still
                    # allowing attending to cached prefix when present.
                    attn_mask = torch.zeros(
                        (1, 1, hidden_states.shape[1], past_len + hidden_states.shape[1]),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
            position_ids = torch.arange(
                past_len, past_len + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
            cache_position = position_ids[0]
            position_embeddings = rotary_emb(hidden_states, position_ids)
            restore_flash_is_causal = None
            attn_module = getattr(layer, "self_attn", None)
            if self._uses_flash_attention_2() and (not causal) and attn_module is not None and hasattr(attn_module, "is_causal"):
                restore_flash_is_causal = bool(getattr(attn_module, "is_causal"))
                setattr(attn_module, "is_causal", False)
            try:
                outputs = layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=cache_obj,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            finally:
                if restore_flash_is_causal is not None:
                    setattr(attn_module, "is_causal", restore_flash_is_causal)
            if not isinstance(outputs, torch.Tensor):
                raise TypeError(
                    "Transformers 4.57 Qwen2DecoderLayer must return a Tensor; "
                    f"layer={idx}, got={type(outputs)}"
                )
            hidden_states = outputs
            present_tuple = self._layer_present_to_tuple(None, idx, cache_obj)
            if present_tuple is None:
                raise RuntimeError(f"Layer {idx} returned empty cache")
            new_cache[idx] = present_tuple

        return hidden_states, new_cache
