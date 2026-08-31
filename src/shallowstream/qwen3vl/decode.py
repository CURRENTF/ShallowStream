"""Qwen3VLDecodeMixin implementation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import _as_bool, _as_int, _full_kv_enabled
from .state import _SelectedCache


class Qwen3VLDecodeMixin:

    def _build_selected_context(
        self,
        selection: Dict[str, Any],
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
        prune_hidden: torch.Tensor,
        input_seq_len: int,
        video_positions: torch.Tensor,
        rope_position_ids: torch.Tensor,
        deepstack_count: int,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Dict[str, Any],
    ]:
        device = video_positions.device
        prefix_positions = torch.arange(0, int(video_positions[0].item()), device=device, dtype=video_positions.dtype)
        suffix_positions = torch.arange(
            int(video_positions[-1].item()) + 1,
            input_seq_len,
            device=device,
            dtype=video_positions.dtype,
        )

        short_frames: List[_FrameKVState] = selection["short"]
        clusters: List[_LongKVCluster] = selection["clusters"]
        if isinstance(selection.get("_compact_prefix_positions"), torch.Tensor):
            prefix_positions = selection["_compact_prefix_positions"].to(device=device, dtype=torch.long)
            suffix_positions = selection["_compact_suffix_positions"].to(device=device, dtype=torch.long)
            prefix_rope_positions = selection["_prefix_rope_positions"].to(device=device, dtype=torch.long)
            suffix_rope_positions = selection["_suffix_rope_positions"].to(device=device, dtype=torch.long)
        else:
            prefix_rope_positions = self._select_rope_positions(rope_position_ids, prefix_positions)
            suffix_rope_positions = self._select_rope_positions(rope_position_ids, suffix_positions)

        hidden_parts: List[torch.Tensor] = []
        visual_mask_parts: List[torch.Tensor] = []
        deepstack_parts: List[List[torch.Tensor]] = [[] for _ in range(int(deepstack_count))]
        deepstack_complete = int(deepstack_count) > 0

        def _append_segment(
            hidden: torch.Tensor,
            is_visual: bool,
            deepstack_embeds: Optional[List[torch.Tensor]] = None,
            visual_mask: Optional[torch.Tensor] = None,
        ) -> None:
            nonlocal deepstack_complete
            hidden = hidden.to(device=prune_hidden.device, dtype=prune_hidden.dtype)
            hidden_parts.append(hidden)
            length = int(hidden.shape[1])
            if visual_mask is None:
                segment_visual_mask = torch.full(
                    (1, length),
                    bool(is_visual),
                    device=prune_hidden.device,
                    dtype=torch.bool,
                )
            else:
                segment_visual_mask = visual_mask.to(device=prune_hidden.device, dtype=torch.bool).view(1, length)
            visual_mask_parts.append(segment_visual_mask)
            if not bool(segment_visual_mask.any().item()) or not deepstack_complete:
                return
            if deepstack_embeds is None or len(deepstack_embeds) < int(deepstack_count):
                deepstack_complete = False
                return
            for layer_idx in range(int(deepstack_count)):
                deepstack_parts[layer_idx].append(
                    deepstack_embeds[layer_idx].to(device=prune_hidden.device, dtype=prune_hidden.dtype)
                )

        _append_segment(self._index_hidden_positions(prune_hidden, prefix_positions), is_visual=False)
        for cluster in clusters:
            _append_segment(cluster.hidden, is_visual=True, deepstack_embeds=cluster.deepstack_embeds)
        for frame in short_frames:
            frame_hidden = self._index_hidden_positions(prune_hidden, frame.context_indices).detach()
            frame_deepstack = self._select_deepstack_visual_embeds(
                token_positions=frame.token_indices,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            context_indices = frame.context_indices.to(device=frame.token_indices.device, dtype=torch.long)
            visual_indices = frame.token_indices.to(device=context_indices.device, dtype=torch.long)
            frame_visual_mask = torch.isin(context_indices, visual_indices).view(1, -1)
            _append_segment(
                frame_hidden,
                is_visual=True,
                deepstack_embeds=frame_deepstack,
                visual_mask=frame_visual_mask,
            )
        _append_segment(self._index_hidden_positions(prune_hidden, suffix_positions), is_visual=False)
        selected_hidden = torch.cat(hidden_parts, dim=1)
        selected_visual_pos_masks = torch.cat(visual_mask_parts, dim=1)
        selected_deepstack_visual_embeds: Optional[List[torch.Tensor]] = None
        if deepstack_complete:
            selected_deepstack_visual_embeds = []
            for layer_idx in range(int(deepstack_count)):
                if deepstack_parts[layer_idx]:
                    selected_deepstack_visual_embeds.append(torch.cat(deepstack_parts[layer_idx], dim=0))
                else:
                    selected_deepstack_visual_embeds.append(
                        prune_hidden.new_empty((0, int(prune_hidden.shape[-1])))
                    )

        selected_positions = self._compose_selected_positions(
            prefix_positions=prefix_rope_positions,
            clusters=clusters,
            short_frames=short_frames,
            suffix_positions=suffix_rope_positions,
            selected_len=int(selected_hidden.shape[1]),
        )

        lower_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            base_k, base_v = self._raw_layer(raw_lower_kv, layer_idx)
            k_parts = [self._index_cache_positions(base_k, prefix_positions)]
            v_parts = [self._index_cache_positions(base_v, prefix_positions)]
            for cluster in clusters:
                k, v = cluster.lower_kv[layer_idx]
                k_parts.append(k.to(device=base_k.device, dtype=base_k.dtype))
                v_parts.append(v.to(device=base_v.device, dtype=base_v.dtype))
            for frame in short_frames:
                k_parts.append(self._index_cache_positions(base_k, frame.context_indices))
                v_parts.append(self._index_cache_positions(base_v, frame.context_indices))
            k_parts.append(self._index_cache_positions(base_k, suffix_positions))
            v_parts.append(self._index_cache_positions(base_v, suffix_positions))
            raw_key = torch.cat(k_parts, dim=2)
            value = torch.cat(v_parts, dim=2)
            lower_cache[layer_idx] = {
                "k": raw_key,
                "v": value,
                "positions": selected_positions.to(device=raw_key.device, dtype=torch.long),
            }

        next_position = int(selected_positions[0].max().item()) + 1 if selected_positions.numel() > 0 else 0

        stats = {
            "prune_layer": _as_int(self.config, "prune_layer"),
            "rope_position_mode": self._rope_position_mode(),
            "deepstack_injection_text_layers": list(range(int(deepstack_count))),
            "prefix_tokens": int(prefix_positions.numel()),
            "suffix_tokens": int(suffix_positions.numel()),
            "long_source_units": int(selection.get("long_source_count", len(selection["long_source"]))),
            "long_cluster_count": len(selection["all_clusters"]),
            "selected_long_clusters": len(clusters),
            "selected_cluster_unit_counts": [cluster.count for cluster in clusters],
            "short_search_candidates": int(
                selection.get("short_search_candidate_count", len(selection["search_candidates"]))
            ),
            "evicted_long_unit_count": int(selection.get("evicted_long_frame_count", 0)),
            "evicted_raw_token_count": int(selection.get("evicted_raw_token_count", 0)),
            "retrieved_seed_units": [frame.frame_id for frame in selection["retrieved_seed"]],
            "retrieved_short_units": [frame.frame_id for frame in selection["retrieved"]],
            "retrieval_expand_prev_units": self._retrieval_expand_prev_units(),
            "retrieval_expand_next_units": self._retrieval_expand_next_units(),
            "retrieval_expand_prev_stride_units": self._retrieval_expand_prev_stride_units(),
            "retrieval_expand_next_stride_units": self._retrieval_expand_next_stride_units(),
            "recent_units": [frame.frame_id for frame in selection["recent"]],
            "selected_short_units": [frame.frame_id for frame in short_frames],
            "selected_context_tokens": int(selected_hidden.shape[1]),
            "selected_visual_tokens": int(selected_visual_pos_masks.sum().item()),
            "selected_deepstack_layers": (
                len(selected_deepstack_visual_embeds)
                if selected_deepstack_visual_embeds is not None
                else 0
            ),
            "persistent_frame_lower_kv_copies": 0,
            "selected_position_min": int(selected_positions[0].min().item()) if selected_positions.numel() else 0,
            "selected_position_max": int(selected_positions[0].max().item()) if selected_positions.numel() else 0,
            "next_position": next_position,
            "use_rekv_sink": _as_bool(self.config, "use_rekv_sink"),
            "rekv_sink_len": int(prefix_positions.numel()),
            "shallow_prefill_local_window_frames": self._local_window_frames(),
            "local_window_units": self._local_window_units(),
        }
        stats.update(selection.get("token_selection_stats") or {})
        return (
            lower_cache,
            selected_hidden,
            selected_positions,
            next_position,
            selected_visual_pos_masks,
            selected_deepstack_visual_embeds,
            stats,
        )

    def _layer_device(self, layer: torch.nn.Module, fallback: torch.device) -> torch.device:
        try:
            return next(layer.parameters()).device
        except StopIteration:
            return fallback

    def _position_ids(self, positions: torch.Tensor, device: torch.device, use_mrope: bool) -> torch.Tensor:
        base = positions.to(device=device, dtype=torch.long)
        if use_mrope:
            if base.dim() == 1:
                return base.view(1, -1)
            if base.dim() == 2 and int(base.shape[0]) == 3:
                return base[:, None, :].contiguous()
            if base.dim() == 3:
                return base
        if base.dim() == 2 and int(base.shape[0]) == 3:
            return base[0].view(1, -1)
        if base.dim() == 3:
            return base[0]
        return base.view(1, -1)

    def _rotary(self, rotary_emb, hidden: torch.Tensor, positions: torch.Tensor):
        pos3 = self._position_ids(positions, hidden.device, use_mrope=True)
        try:
            return rotary_emb(hidden, pos3), self._position_ids(positions, hidden.device, use_mrope=False)
        except Exception:
            pos2 = self._position_ids(positions, hidden.device, use_mrope=False)
            try:
                return rotary_emb(hidden, pos2), pos2
            except TypeError:
                return rotary_emb(pos2), pos2

    def _causal_mask(
        self,
        hidden: torch.Tensor,
        query_len: int,
        past_len: int,
    ) -> Optional[torch.Tensor]:
        if query_len <= 1:
            return None
        key_len = past_len + query_len
        query_ids = torch.arange(query_len, device=hidden.device).view(query_len, 1)
        key_ids = torch.arange(key_len, device=hidden.device).view(1, key_len)
        blocked = key_ids > (past_len + query_ids)
        mask = torch.zeros((query_len, key_len), device=hidden.device, dtype=hidden.dtype)
        mask = mask.masked_fill(blocked, torch.finfo(hidden.dtype).min)
        return mask.view(1, 1, query_len, key_len)

    def _run_layers(
        self,
        hidden: torch.Tensor,
        layers,
        rotary_emb,
        start_layer: int,
        end_layer: int,
        cache: _SelectedCache,
        positions: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        for layer_idx in range(start_layer, end_layer):
            layer = layers[layer_idx]
            layer_device = self._layer_device(layer, hidden.device)
            hidden = hidden.to(layer_device)
            length = int(hidden.shape[1])
            layer_positions = positions.to(device=layer_device, dtype=torch.long)
            if int(layer_positions.shape[-1]) != length:
                raise RuntimeError(f"position count {int(layer_positions.shape[-1])} != hidden length {length}.")
            position_embeddings, position_ids = self._rotary(rotary_emb, hidden, layer_positions)
            past_len = cache.get_layer_seq_length(layer_idx)
            # HF FlashAttention2 handles causal masking internally. Passing a
            # 4D additive mask can force a slower/incompatible path.
            attention_mask = None if self._uses_flash_attention_2() else self._causal_mask(hidden, length, past_len)
            cache_position = layer_positions
            kwargs = {
                "hidden_states": hidden,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": cache,
                "use_cache": True,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
            }
            try:
                layer_out = layer(**kwargs)
            except TypeError:
                kwargs.pop("use_cache", None)
                kwargs.pop("position_ids", None)
                layer_out = layer(**kwargs)
            hidden = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden = self._deepstack_process(
                    hidden_states=hidden,
                    visual_pos_masks=visual_pos_masks,
                    visual_embeds=deepstack_visual_embeds[layer_idx],
                )
        return hidden

    def _eos_token_ids(self) -> List[int]:
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if eos is None:
            return []
        if isinstance(eos, (list, tuple, set)):
            return [int(x) for x in eos if x is not None]
        return [int(eos)]

    def _decode_from_selected_cache(
        self,
        selected_lower_cache: Dict[int, Dict[str, torch.Tensor]],
        selected_hidden: torch.Tensor,
        selected_positions: torch.Tensor,
        next_position: int,
        selected_visual_pos_masks: Optional[torch.Tensor],
        selected_deepstack_visual_embeds: Optional[List[torch.Tensor]],
        layers,
        rotary_emb,
        norm,
        lm_head,
        embed_tokens,
        prune_layer: int,
        sink_len: int,
        local_window_tokens: int,
    ) -> str:
        sync_latency = _as_bool(self.config, "latency_sync_cuda")
        if sync_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_start_time = time.perf_counter()
        upper_cache = _SelectedCache()
        upper_input = (
            selected_hidden[:, -1:, :]
            if _full_kv_enabled(self.config) and prune_layer == len(layers)
            else selected_hidden
        )
        upper_positions = (
            selected_positions[..., -1:]
            if int(upper_input.shape[1]) == 1 and int(selected_positions.shape[-1]) > 1
            else selected_positions
        )
        hidden = self._run_layers(
            hidden=upper_input,
            layers=layers,
            rotary_emb=rotary_emb,
            start_layer=prune_layer,
            end_layer=len(layers),
            cache=upper_cache,
            positions=upper_positions,
            visual_pos_masks=selected_visual_pos_masks,
            deepstack_visual_embeds=selected_deepstack_visual_embeds,
        )

        norm_device = self._layer_device(norm, hidden.device)
        head_device = self._layer_device(lm_head, norm_device)
        logits = lm_head(norm(hidden.to(norm_device)).to(head_device)[:, -1:, :])[:, -1, :]
        generated: List[int] = []
        max_new_tokens = _as_int(self.config, "max_new_tokens")
        eos_ids = set(self._eos_token_ids())
        do_sample = _as_bool(self.config, "do_sample")
        temperature = max(float(self.config.get("decode_temperature", 1.0)), 1e-6)
        first_token_time = None

        for step in range(max_new_tokens):
            if do_sample:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if first_token_time is None:
                if sync_latency and torch.cuda.is_available():
                    torch.cuda.synchronize()
                first_token_time = time.perf_counter()
            if token_id in eos_ids and not _as_bool(self.config, "force_exact_new_tokens"):
                break

            embed_device = self._layer_device(embed_tokens, next_token.device)
            token_hidden = embed_tokens(next_token.to(embed_device))
            token_position = self._scalar_rope_position(next_position + step, embed_device)
            token_hidden, selected_lower_cache = self._forward_lower_layers_raw(
                hidden_states=token_hidden,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=prune_layer,
                past_raw_kv=selected_lower_cache,
                positions=token_position,
                update_cache=True,
                consume_past_cache=_full_kv_enabled(self.config),
                sink_len=int(sink_len),
                local_window_tokens=int(local_window_tokens),
            )
            token_hidden = self._run_layers(
                hidden=token_hidden,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=prune_layer,
                end_layer=len(layers),
                cache=upper_cache,
                positions=token_position,
            )
            norm_device = self._layer_device(norm, token_hidden.device)
            head_device = self._layer_device(lm_head, norm_device)
            logits = lm_head(norm(token_hidden.to(norm_device)).to(head_device)[:, -1:, :])[:, -1, :]

        if sync_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._last_decode_observation = {
            "generation_start_time": generation_start_time,
            "first_token_time": first_token_time,
            "generation_end_time": time.perf_counter(),
            "generated_tokens": len(generated),
        }
        if not generated:
            return ""
        text = self.processor.batch_decode(
            [generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text[0].strip() if text else ""
