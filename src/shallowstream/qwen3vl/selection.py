"""Qwen3VLSelectionMixin implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from src.shallowstream.common import select_temporal_retrieval_ids

from .config import _as_bool, _as_float, _as_int, _retrieval_expansion_strategy
from .state import _FrameKVState, _LongKVCluster


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as _qwen3_apply_rope
except Exception:  # Keep import-time compatibility with machines that have older transformers.
    _qwen3_apply_rope = None


class Qwen3VLSelectionMixin:

    def _retrieval_vote_query_indices(
        self,
        text_inputs: Dict[str, Any],
        prompt: str,
    ) -> List[int]:
        if self._retrieval_vote_query_token_mode() == "prompt_last":
            input_ids = text_inputs.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                sequence_length = int(input_ids.shape[-1])
            elif isinstance(input_ids, (list, tuple)) and input_ids:
                first = input_ids[0]
                sequence_length = (
                    len(first) if isinstance(first, (list, tuple)) else len(input_ids)
                )
            else:
                raise RuntimeError(
                    "prompt-last token voting requires input_ids in the model text inputs"
                )
            if sequence_length <= 0:
                raise RuntimeError("prompt-last token voting received an empty prompt")
            return [sequence_length - 1]
        return self._attention_distribution_query_indices(text_inputs, prompt)

    def _recent_retrieval_unit_score_observation(
        self,
        *,
        recent_frames: Sequence[_FrameKVState],
        query_vec: torch.Tensor,
        query_layer: int,
        slot_count: int,
    ) -> Optional[Dict[str, Any]]:
        if not _as_bool(self.config, "observe_recent_retrieval_unit_scores"):
            return None
        if slot_count <= 0:
            raise ValueError(
                "observe_recent_retrieval_unit_scores requires retrieval_recent_units > 0"
            )

        scored_units: List[Dict[str, Any]] = []
        for frame in recent_frames[-slot_count:]:
            scored_units.append(
                {
                    "frame_id": int(frame.frame_id),
                    "sample_index": int(frame.sample_index),
                    "timestamp": float(frame.timestamp),
                    "source_sample_indices": [
                        int(index) for index in (frame.source_sample_indices or [])
                    ],
                    "source_timestamps": [
                        float(timestamp) for timestamp in (frame.source_timestamps or [])
                    ],
                    "score": float(
                        torch.dot(frame.key_vec.float(), query_vec.float()).item()
                    ),
                }
            )

        missing = slot_count - len(scored_units)
        unit_slots: List[Optional[Dict[str, Any]]] = [None] * missing + scored_units
        return {
            "schema_version": 1,
            "metric": "shallow_qk_cosine",
            "representation": "normalized_temporal_unit_k_dot_normalized_question_q",
            "granularity": "qwen_temporal_unit",
            "query_layer_index": int(query_layer),
            "query_layer_depth": int(query_layer) + 1,
            "video_temporal_patch_size": int(
                self.config.get("video_temporal_patch_size", 2)
            ),
            "slot_count": int(slot_count),
            "available_unit_count": len(scored_units),
            "slot_order": "oldest_to_latest_right_aligned",
            "scores": [
                None if unit is None else float(unit["score"])
                for unit in unit_slots
            ],
            "units": unit_slots,
        }

    def _subset_frame_state(
        self,
        frame: _FrameKVState,
        selected_token_indices: torch.Tensor,
    ) -> _FrameKVState:
        selected_token_indices = selected_token_indices.detach().to(
            device=frame.token_indices.device,
            dtype=torch.long,
        )
        if selected_token_indices.numel() <= 0:
            raise ValueError("selected_token_indices must be non-empty")

        frame_tokens = frame.token_indices.to(device=selected_token_indices.device, dtype=torch.long)
        token_keep_mask = torch.isin(frame_tokens, selected_token_indices)
        if int(token_keep_mask.sum().item()) != int(selected_token_indices.numel()):
            missing = selected_token_indices[~torch.isin(selected_token_indices, frame_tokens)]
            raise RuntimeError(
                f"token-level selection referenced tokens outside frame={frame.frame_id}: "
                f"{missing[:8].detach().cpu().tolist()}"
            )

        context_indices = frame.context_indices.to(device=selected_token_indices.device, dtype=torch.long)
        context_visual_mask = torch.isin(context_indices, frame_tokens)
        context_keep_mask = (~context_visual_mask) | torch.isin(context_indices, selected_token_indices)
        subset_context_indices = context_indices[context_keep_mask].detach().clone()

        context_positions = frame.context_positions.to(device=context_keep_mask.device)
        if context_positions.dim() == 1:
            subset_context_positions = context_positions[context_keep_mask].detach().clone()
        else:
            subset_context_positions = context_positions.index_select(
                -1,
                torch.nonzero(context_keep_mask, as_tuple=False).flatten().to(context_positions.device),
            ).detach().clone()

        positions = frame.positions.to(device=token_keep_mask.device)
        if positions.dim() == 1:
            subset_positions = positions[token_keep_mask].detach().clone()
        else:
            subset_positions = positions.index_select(
                -1,
                torch.nonzero(token_keep_mask, as_tuple=False).flatten().to(positions.device),
            ).detach().clone()

        visual_embeds = None
        if isinstance(frame.visual_embeds, torch.Tensor):
            embeds = frame.visual_embeds.detach()
            visual_embeds = embeds.index_select(
                0,
                torch.nonzero(token_keep_mask, as_tuple=False).flatten().to(embeds.device),
            ).detach().clone()
        deepstack_embeds = None
        if isinstance(frame.deepstack_embeds, list):
            keep = torch.nonzero(token_keep_mask, as_tuple=False).flatten()
            deepstack_embeds = [
                embeds.detach().index_select(0, keep.to(embeds.device)).clone()
                for embeds in frame.deepstack_embeds
            ]

        return _FrameKVState(
            frame_id=frame.frame_id,
            sample_index=frame.sample_index,
            timestamp=frame.timestamp,
            token_indices=frame_tokens[token_keep_mask].detach().clone(),
            context_indices=subset_context_indices,
            context_positions=subset_context_positions,
            positions=subset_positions,
            key_vec=frame.key_vec,
            visual_embeds=visual_embeds,
            deepstack_embeds=deepstack_embeds,
            grid_thw=self._grid_for_visual_token_count(int(selected_token_indices.numel()), frame.grid_thw),
            image=None,
        )

    def _select_token_level_short_frames(
        self,
        *,
        candidate_frames: Sequence[_FrameKVState],
        selected_unit_frames: Sequence[_FrameKVState],
        recent_frames: Sequence[_FrameKVState],
        key: torch.Tensor,
        query_vec: torch.Tensor,
    ) -> Tuple[List[_FrameKVState], List[_FrameKVState], Dict[str, Any]]:
        recent_by_id = {int(frame.frame_id): frame for frame in recent_frames}
        unit_by_id = {int(frame.frame_id): frame for frame in selected_unit_frames}
        unit_context_budget = sum(int(frame.context_indices.numel()) for frame in unit_by_id.values())
        recent_context_budget = sum(
            int(frame.context_indices.numel())
            for frame_id, frame in recent_by_id.items()
            if frame_id in unit_by_id
        )
        retrieval_context_budget = max(unit_context_budget - recent_context_budget, 0)
        if retrieval_context_budget <= 0 or not candidate_frames:
            return [], list(recent_frames), {
                "retrieval_selection_granularity": "token",
                "token_context_budget": retrieval_context_budget,
                "token_context_used": 0,
                "token_visual_budget_equivalent": sum(int(frame.token_indices.numel()) for frame in unit_by_id.values()),
                "token_selected_visual_tokens": 0,
                "token_selected_frame_count": 0,
            }

        scored_tokens: List[Tuple[float, int, int]] = []
        frame_by_id = {int(frame.frame_id): frame for frame in candidate_frames}
        reverse = self._retrieval_score_reverse()
        for frame in candidate_frames:
            positions = frame.token_indices.to(device=key.device, dtype=torch.long)
            if positions.numel() <= 0:
                continue
            token_vectors = self._normalized_key_vectors(key, positions)
            scores = torch.matmul(token_vectors, query_vec.to(device=token_vectors.device, dtype=token_vectors.dtype))
            for local_idx, score in enumerate(scores.detach().cpu().tolist()):
                scored_tokens.append((float(score), int(frame.frame_id), int(local_idx)))
        scored_tokens.sort(key=lambda item: item[0], reverse=reverse)

        selected_local: Dict[int, List[int]] = {}
        used_context = 0
        for _score, frame_id, local_idx in scored_tokens:
            frame = frame_by_id.get(frame_id)
            if frame is None:
                continue
            locals_for_frame = selected_local.get(frame_id, [])
            if local_idx in locals_for_frame:
                continue
            if locals_for_frame:
                cost = 1
            else:
                context_indices = frame.context_indices.to(device=frame.token_indices.device, dtype=torch.long)
                frame_tokens = frame.token_indices.to(device=context_indices.device, dtype=torch.long)
                nonvisual_count = int((~torch.isin(context_indices, frame_tokens)).sum().item())
                cost = nonvisual_count + 1
            if used_context + cost > retrieval_context_budget:
                continue
            if frame_id not in selected_local:
                selected_local[frame_id] = locals_for_frame
            locals_for_frame.append(local_idx)
            used_context += cost
            if used_context >= retrieval_context_budget:
                break

        token_frames: List[_FrameKVState] = []
        for frame_id in sorted(selected_local):
            frame = frame_by_id[frame_id]
            local_indices = sorted(set(selected_local[frame_id]))
            local_tensor = torch.tensor(local_indices, device=frame.token_indices.device, dtype=torch.long)
            selected_tokens = frame.token_indices.index_select(0, local_tensor)
            token_frames.append(self._subset_frame_state(frame, selected_tokens))

        selected_short = sorted(
            {frame.frame_id: frame for frame in token_frames + list(recent_frames)}.values(),
            key=lambda frame: frame.frame_id,
        )
        stats = {
            "retrieval_selection_granularity": "token",
            "token_context_budget": int(retrieval_context_budget),
            "token_context_used": int(used_context),
            "token_unit_context_budget": int(unit_context_budget),
            "token_recent_context_budget": int(recent_context_budget),
            "token_visual_budget_equivalent": sum(int(frame.token_indices.numel()) for frame in unit_by_id.values()),
            "token_selected_visual_tokens": sum(int(frame.token_indices.numel()) for frame in token_frames),
            "token_selected_frame_count": len(token_frames),
            "token_selected_units": [int(frame.frame_id) for frame in token_frames],
        }
        return token_frames, selected_short, stats

    def _extract_frame_lower_kv(
        self,
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
        token_indices: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        lower_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            key, value = self._raw_layer(raw_lower_kv, layer_idx)
            lower_kv[layer_idx] = (
                self._index_cache_positions(key, token_indices).detach(),
                self._index_cache_positions(value, token_indices).detach(),
            )
        return lower_kv

    def _select_deepstack_visual_embeds(
        self,
        token_positions: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        if visual_pos_masks is None or not isinstance(deepstack_visual_embeds, list) or not deepstack_visual_embeds:
            return None
        if token_positions.numel() == 0:
            return []

        positions = token_positions.detach().long().view(-1)
        mask = visual_pos_masks.detach()
        if mask.dim() == 2:
            mask = mask[0]
        elif mask.dim() > 2:
            mask = mask.reshape(-1)
        mask = mask.to(device=positions.device, dtype=torch.bool)

        visual_positions = torch.nonzero(mask, as_tuple=False).flatten().long()
        if visual_positions.numel() == 0:
            return None

        max_len = max(int(mask.numel()), int(positions.max().item()) + 1)
        rank_map = torch.full((max_len,), -1, device=positions.device, dtype=torch.long)
        rank_map[visual_positions] = torch.arange(visual_positions.numel(), device=positions.device)
        ranks = rank_map[positions]
        if torch.any(ranks < 0):
            missing = positions[ranks < 0][:8].detach().cpu().tolist()
            raise RuntimeError(f"Could not map video token positions to Qwen3 DeepStack ranks: {missing}")

        selected: List[torch.Tensor] = []
        for layer_idx, embeds in enumerate(deepstack_visual_embeds):
            if not isinstance(embeds, torch.Tensor):
                return None
            layer_embeds = embeds.detach()
            if layer_embeds.dim() == 3 and int(layer_embeds.shape[0]) == 1:
                layer_embeds = layer_embeds[0]
            if int(ranks.max().item()) >= int(layer_embeds.shape[0]):
                raise RuntimeError(
                    f"Qwen3 DeepStack layer {layer_idx} has {int(layer_embeds.shape[0])} visual tokens, "
                    f"but selected rank reaches {int(ranks.max().item())}."
                )
            selected.append(layer_embeds.index_select(0, ranks.to(layer_embeds.device)).clone())
        return selected

    def _build_frame_states(
        self,
        frames: Sequence[SampledFrame],
        frame_positions: Sequence[torch.Tensor],
        rope_position_ids: torch.Tensor,
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        query_layer: int,
        input_embeds: Optional[torch.Tensor] = None,
        unit_grid_rows: Optional[Sequence[torch.Tensor]] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> List[_FrameKVState]:
        key_for_query, _ = self._raw_layer(raw_lower_kv, query_layer)
        states: List[_FrameKVState] = []
        frame_context_positions = self._split_video_context_positions(frame_positions)
        for frame_id, (frame, positions) in enumerate(zip(frames, frame_positions)):
            context_indices = frame_context_positions[frame_id]
            frame_rope_positions = self._select_rope_positions(rope_position_ids, positions)
            frame_context_rope_positions = self._select_rope_positions(rope_position_ids, context_indices)
            frame_visual_embeds = None
            if isinstance(input_embeds, torch.Tensor):
                frame_visual_embeds = self._index_hidden_positions(input_embeds, positions)[0].detach().to(
                    device="cpu",
                    dtype=torch.bfloat16,
                )
            frame_deepstack = None
            if self._selected_generate_mode() == "simple_prompt":
                frame_deepstack = self._select_deepstack_visual_embeds(
                    token_positions=positions,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            frame_grid = None
            if unit_grid_rows is not None and frame_id < len(unit_grid_rows):
                frame_grid = unit_grid_rows[frame_id].detach().cpu().long()
            states.append(
                _FrameKVState(
                    frame_id=frame_id,
                    sample_index=frame.index,
                    timestamp=frame.timestamp,
                    token_indices=positions.detach().clone(),
                    context_indices=context_indices.detach().clone(),
                    context_positions=frame_context_rope_positions.detach().clone(),
                    positions=frame_rope_positions.detach().clone(),
                    key_vec=self._normalized_key_vector(key_for_query, positions).detach(),
                    visual_embeds=frame_visual_embeds,
                    deepstack_embeds=(
                        [emb.detach().to(device="cpu", dtype=torch.bfloat16) for emb in frame_deepstack]
                        if frame_deepstack is not None
                        else None
                    ),
                    grid_thw=frame_grid,
                    image=frame.image,
                    source_images=[
                        source.image
                        for source in (frame.source_frames or [frame])
                        if isinstance(source.image, Image.Image)
                    ],
                    source_sample_indices=[
                        int(source.index) for source in (frame.source_frames or [frame])
                    ],
                    source_timestamps=[
                        float(source.timestamp) for source in (frame.source_frames or [frame])
                    ],
                )
            )
        return states

    def _query_vector(
        self,
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        query_layer: int,
        video_positions: torch.Tensor,
    ) -> torch.Tensor:
        layer_cache = raw_lower_kv.get(query_layer)
        if not isinstance(layer_cache, dict) or not isinstance(layer_cache.get("q"), torch.Tensor):
            raise RuntimeError(
                f"Missing raw Q capture for retrieval query at lower layer {query_layer}. "
                "The first lower prefill must pass capture_q_layers={prune_layer - 1}."
            )
        query = layer_cache["q"]
        key, _ = self._raw_layer(raw_lower_kv, query_layer)
        suffix_start = int(video_positions[-1].item()) + 1
        query_len = int(query.shape[2])
        if suffix_start >= query_len:
            raise RuntimeError("Qwen3-VL full prompt has no question tokens after the video")
        positions = torch.arange(
            suffix_start,
            query_len,
            device=query.device,
            dtype=torch.long,
        )
        return self._normalized_query_vector(query, positions, key_head_count=int(key.shape[1])).detach()

    def _select_shallow_layer_token_vote_frames(
        self,
        *,
        candidate_frames: Sequence[_FrameKVState],
        q_cache: Dict[int, Dict[str, torch.Tensor]],
        k_cache: Dict[int, Dict[str, torch.Tensor]],
        query_indices: Sequence[int],
        rotary_emb,
        layers: Sequence[Any],
        layer_indices: Sequence[int],
        frame_budget: int,
        token_topk: int,
    ) -> Tuple[
        List[_FrameKVState],
        List[Tuple[float, _FrameKVState]],
        Dict[str, Any],
    ]:
        frames = list(candidate_frames)
        if not frames or frame_budget <= 0:
            return [], [], {
                "retrieval_score_strategy": "shallow_layer_token_vote",
                "retrieval_vote_selected_unit_ids": [],
            }
        indices = [int(index) for index in query_indices]
        if not indices:
            raise ValueError("shallow-layer token voting requires at least one query token")
        query_token_mode = self._retrieval_vote_query_token_mode()

        visual_indices = torch.cat(
            [frame.token_indices.detach().long().view(-1) for frame in frames],
            dim=0,
        )
        token_frame_offsets = torch.cat(
            [
                torch.full(
                    (int(frame.token_indices.numel()),),
                    frame_offset,
                    dtype=torch.long,
                    device=visual_indices.device,
                )
                for frame_offset, frame in enumerate(frames)
            ],
            dim=0,
        )
        frame_votes = torch.zeros(len(frames), dtype=torch.long)
        frame_attention = torch.zeros(len(frames), dtype=torch.float64)
        per_layer_selected_unit_ids: Dict[str, List[int]] = {}
        effective_topk = min(int(token_topk), int(visual_indices.numel()))

        for layer_idx in layer_indices:
            q_entry = q_cache.get(int(layer_idx), {})
            k_entry = k_cache.get(int(layer_idx), {})
            q_raw = q_entry.get("q")
            k_raw = k_entry.get("k")
            q_positions = q_entry.get("positions")
            k_positions = k_entry.get("positions")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (q_raw, k_raw, q_positions, k_positions)
            ):
                raise RuntimeError(
                    "shallow-layer token voting is missing Q/K/position capture at "
                    f"layer {layer_idx}"
                )
            if max(indices) >= int(q_raw.shape[2]):
                raise IndexError(
                    f"question token index exceeds layer-{layer_idx} Q length"
                )

            query_index = torch.tensor(indices, device=q_raw.device, dtype=torch.long)
            visual_index = visual_indices.to(device=k_raw.device, dtype=torch.long)
            q_selected = q_raw.index_select(2, query_index)
            k_selected = k_raw.index_select(2, visual_index)
            q_rope_positions = q_positions.index_select(
                -1, query_index.to(device=q_positions.device)
            )
            k_rope_positions = k_positions.index_select(
                -1, visual_index.to(device=k_positions.device)
            )
            target_device = q_selected.device
            k_selected = k_selected.to(device=target_device, dtype=q_selected.dtype)
            q_rope_positions = q_rope_positions.to(device=target_device, dtype=torch.long)
            k_rope_positions = k_rope_positions.to(device=target_device, dtype=torch.long)
            q_rot = self._apply_rope_to_key(q_selected, q_rope_positions, rotary_emb)
            k_rot = self._apply_rope_to_key(k_selected, k_rope_positions, rotary_emb)
            if int(q_rot.shape[1]) != int(k_rot.shape[1]):
                if int(q_rot.shape[1]) % int(k_rot.shape[1]) != 0:
                    raise RuntimeError(
                        "Q/KV head mismatch for shallow-layer token voting: "
                        f"q={int(q_rot.shape[1])}, kv={int(k_rot.shape[1])}"
                    )
                k_rot = self._repeat_kv(
                    k_rot,
                    int(q_rot.shape[1]) // int(k_rot.shape[1]),
                )
            attention_module = self._attention_module(layers[int(layer_idx)])
            scaling = float(
                getattr(attention_module, "scaling", q_rot.shape[-1] ** -0.5)
            )
            logits = torch.matmul(
                q_rot.float(),
                k_rot.float().transpose(2, 3),
            ).mul_(scaling)[0]
            token_attention = logits.softmax(dim=-1).mean(dim=(0, 1))
            selected_offsets = torch.topk(
                token_attention,
                k=effective_topk,
                largest=True,
                sorted=True,
            ).indices
            selected_frame_offsets = token_frame_offsets.index_select(
                0, selected_offsets.to(device=token_frame_offsets.device)
            ).cpu()
            selected_attention = token_attention.index_select(0, selected_offsets).double().cpu()
            frame_votes.scatter_add_(
                0,
                selected_frame_offsets,
                torch.ones_like(selected_frame_offsets),
            )
            frame_attention.scatter_add_(0, selected_frame_offsets, selected_attention)
            per_layer_selected_unit_ids[str(int(layer_idx))] = [
                int(frames[offset].frame_id) for offset in selected_frame_offsets.tolist()
            ]

        ranked_offsets = sorted(
            range(len(frames)),
            key=lambda offset: (
                -int(frame_votes[offset].item()),
                -float(frame_attention[offset].item()),
                int(frames[offset].frame_id),
            ),
        )
        diversity_mode = self._retrieval_vote_diversity_mode()
        diversity_pool_size = min(
            len(ranked_offsets),
            int(frame_budget) * self._retrieval_vote_diversity_pool_multiplier(),
        )
        if diversity_mode == "divprune_maxmin":
            selected_offsets = self._select_divprune_frame_offsets(
                frames=frames,
                ranked_offsets=ranked_offsets[:diversity_pool_size],
                frame_budget=frame_budget,
            )
        else:
            selected_offsets = ranked_offsets[: min(int(frame_budget), len(frames))]
        selected = [frames[offset] for offset in selected_offsets]
        scored = [
            (float(frame_votes[offset].item()), frames[offset])
            for offset in ranked_offsets
        ]
        stats = {
            "retrieval_score_strategy": "shallow_layer_token_vote",
            "retrieval_vote_layer_indices": [int(index) for index in layer_indices],
            "retrieval_vote_topk_tokens_per_layer": int(effective_topk),
            "retrieval_vote_query_token_mode": query_token_mode,
            "retrieval_vote_query_token_count": len(indices),
            "retrieval_vote_selected_unit_ids": [int(frame.frame_id) for frame in selected],
            "retrieval_vote_diversity_mode": diversity_mode,
            "retrieval_vote_diversity_pool_size": int(diversity_pool_size),
            "retrieval_vote_unit_scores": [
                {
                    "frame_id": int(frames[offset].frame_id),
                    "votes": int(frame_votes[offset].item()),
                    "attention_tiebreak": float(frame_attention[offset].item()),
                }
                for offset in ranked_offsets
            ],
            "retrieval_vote_layer_selected_unit_ids": per_layer_selected_unit_ids,
        }
        return selected, scored, stats

    def _select_divprune_frame_offsets(
        self,
        *,
        frames: Sequence[_FrameKVState],
        ranked_offsets: Sequence[int],
        frame_budget: int,
    ) -> List[int]:
        pool = [int(offset) for offset in ranked_offsets]
        target = min(max(int(frame_budget), 0), len(pool))
        if target <= 1:
            return pool[:target]
        representations = torch.stack(
            [frames[offset].key_vec.detach().float().cpu() for offset in pool],
            dim=0,
        )
        representations = torch.nn.functional.normalize(representations, dim=-1)
        distances = 1.0 - representations @ representations.transpose(0, 1)
        distances.fill_diagonal_(-1.0)

        pair_candidates = []
        for left in range(len(pool)):
            for right in range(left + 1, len(pool)):
                pair_candidates.append(
                    (
                        float(distances[left, right].item()),
                        -(left + right),
                        -left,
                        left,
                        right,
                    )
                )
        _distance, _rank_sum, _left_rank, left, right = max(pair_candidates)
        selected_local = [left, right]
        remaining = set(range(len(pool))) - set(selected_local)
        while len(selected_local) < target:
            best = max(
                remaining,
                key=lambda candidate: (
                    float(distances[candidate, selected_local].min().item()),
                    -candidate,
                ),
            )
            selected_local.append(best)
            remaining.remove(best)
        return [pool[index] for index in selected_local]

    def _select_memory(
        self,
        frame_states: Sequence[_FrameKVState],
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
        prune_hidden: torch.Tensor,
        query_layer: int,
        video_positions: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        prompt: str,
        probe_query_vecs: Optional[Dict[str, torch.Tensor]] = None,
        attention_distribution_layer: Optional[int] = None,
        attention_distribution_features: Optional[Dict[str, float]] = None,
        attention_distribution_observation_features: Optional[Dict[int, Dict[str, float]]] = None,
        token_vote_query_indices: Optional[Sequence[int]] = None,
        rotary_emb=None,
        layers: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        total_frames = len(frame_states)
        recent_n = self._retrieval_recent_units()
        topk = self._retrieval_topk_units()
        search_n = self._retrieval_search_window_units()

        recent_start = max(total_frames - recent_n, 0) if recent_n > 0 else total_frames
        search_end = recent_start
        search_start = max(search_end - search_n, 0) if search_n > 0 else search_end
        long_source = list(frame_states[:search_start])
        search_candidates = list(frame_states[search_start:search_end])
        recent_frames = list(frame_states[recent_start:])

        query_vec = self._query_vector(
            raw_lower_kv,
            query_layer,
            video_positions,
        )
        recent_unit_score_observation = self._recent_retrieval_unit_score_observation(
            recent_frames=recent_frames,
            query_vec=query_vec,
            query_layer=query_layer,
            slot_count=recent_n,
        )

        retrieved: List[_FrameKVState] = []
        retrieved_seed: List[_FrameKVState] = []
        retrieval_reference_expanded_unit_ids: List[int] = []
        scored_short: List[Tuple[float, _FrameKVState]] = []
        vote_stats: Dict[str, Any] = {}
        if (
            self._evidence_retrieval_backend() == "shallow"
            and topk > 0
            and search_candidates
        ):
            if self._retrieval_score_strategy() == "shallow_layer_token_vote":
                if rotary_emb is None or layers is None:
                    raise RuntimeError("shallow-layer token voting requires model layers and RoPE")
                layer_indices = list(
                    range(self._retrieval_vote_layer_start(), int(query_layer) + 1)
                )
                retrieved_seed, scored_short, vote_stats = (
                    self._select_shallow_layer_token_vote_frames(
                        candidate_frames=search_candidates,
                        q_cache=raw_lower_kv,
                        k_cache=raw_lower_kv,
                        query_indices=list(token_vote_query_indices or []),
                        rotary_emb=rotary_emb,
                        layers=layers,
                        layer_indices=layer_indices,
                        frame_budget=topk,
                        token_topk=self._retrieval_vote_topk_tokens_per_layer(),
                    )
                )
            else:
                scored_short = [
                    (float(torch.dot(frame.key_vec.float(), query_vec.float()).item()), frame)
                    for frame in search_candidates
                ]
                reverse = self._retrieval_score_reverse()
                scored_short.sort(key=lambda item: item[0], reverse=reverse)
                retrieved_seed = [frame for _score, frame in scored_short[:topk]]

            expand_prev = self._retrieval_expand_prev_units()
            expand_next = self._retrieval_expand_next_units()
            stride_prev = self._retrieval_expand_prev_stride_units()
            stride_next = self._retrieval_expand_next_stride_units()

            candidate_frames = list(search_candidates) + list(recent_frames)
            candidate_by_id = {frame.frame_id: frame for frame in candidate_frames}
            if candidate_by_id:
                (
                    selected_ids,
                    retrieval_reference_expanded_unit_ids,
                ) = select_temporal_retrieval_ids(
                    (frame.frame_id for frame in retrieved_seed),
                    (frame.frame_id for _score, frame in scored_short),
                    candidate_by_id,
                    previous=expand_prev,
                    following=expand_next,
                    previous_stride=stride_prev,
                    following_stride=stride_next,
                    strategy=_retrieval_expansion_strategy(self.config),
                )
                retrieved = [candidate_by_id[fid] for fid in selected_ids]

        selected_short = sorted({frame.frame_id: frame for frame in retrieved + recent_frames}.values(), key=lambda x: x.frame_id)
        token_selection_stats: Dict[str, Any] = {"retrieval_selection_granularity": "unit"}
        token_selection_stats.update(vote_stats)
        if self._retrieval_selection_granularity() == "token":
            token_retrieved, selected_short, token_selection_stats = self._select_token_level_short_frames(
                candidate_frames=search_candidates,
                selected_unit_frames=selected_short,
                recent_frames=recent_frames,
                key=key,
                query_vec=query_vec,
            )
            retrieved = token_retrieved
        clusters = []
        if _as_bool(self.config, "long_cluster_enabled"):
            clusters = self._cluster_long_kv(
                frames=long_source,
                raw_lower_kv=raw_lower_kv,
                lower_layer_ids=lower_layer_ids,
                prune_hidden=prune_hidden,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        selected_clusters = self._retrieve_long_kv(clusters, query_vec)
        selection = {
            "query_vec": query_vec,
            "short": selected_short,
            "clusters": selected_clusters,
            "all_clusters": clusters,
            "retrieved": retrieved,
            "retrieved_seed": retrieved_seed,
            "retrieval_reference_expanded_unit_ids": retrieval_reference_expanded_unit_ids,
            "recent": recent_frames,
            "search_candidates": search_candidates,
            "long_source": long_source,
            "short_scores": scored_short,
            "token_selection_stats": token_selection_stats,
            "probe_query_vecs": dict(probe_query_vecs or {}),
            "attention_distribution_layer": attention_distribution_layer,
            "attention_distribution_features": attention_distribution_features,
            "attention_distribution_observation_features": attention_distribution_observation_features,
        }
        if recent_unit_score_observation is not None:
            selection["recent_retrieval_unit_scores"] = recent_unit_score_observation
        return self._apply_task_gate(selection, prompt=prompt, source="full_prefill")

    def _cluster_long_kv(
        self,
        frames: Sequence[_FrameKVState],
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
        prune_hidden: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> List[_LongKVCluster]:
        threshold = _as_float(self.config, "cluster_threshold")
        clusters: List[_LongKVCluster] = []
        for frame in frames:
            frame_lower_kv = self._extract_frame_lower_kv(raw_lower_kv, lower_layer_ids, frame.token_indices)
            frame_hidden = self._index_hidden_positions(prune_hidden, frame.token_indices).detach()
            frame_deepstack = self._select_deepstack_visual_embeds(
                token_positions=frame.token_indices,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            if not clusters:
                clusters.append(
                    _LongKVCluster.from_frame(
                        frame,
                        frame_lower_kv,
                        frame_hidden,
                        frame_deepstack,
                    )
                )
                continue
            latest = clusters[-1]
            similarity = float(torch.dot(latest.key_vec.float(), frame.key_vec.float()).item())
            if similarity < threshold:
                clusters.append(
                    _LongKVCluster.from_frame(
                        frame,
                        frame_lower_kv,
                        frame_hidden,
                        frame_deepstack,
                    )
                )
            else:
                latest.merge(
                    frame,
                    frame_lower_kv,
                    frame_hidden,
                    frame_deepstack,
                )
        return clusters

    def _retrieve_long_kv(self, clusters: Sequence[_LongKVCluster], query_vec: torch.Tensor) -> List[_LongKVCluster]:
        topk = max(_as_int(self.config, "long_cluster_topk"), 0)
        if topk <= 0 or not clusters:
            return []
        scored = [
            (float(torch.dot(cluster.key_vec.float(), query_vec.float()).item()), cluster)
            for cluster in clusters
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [cluster for _score, cluster in scored[:topk]]
        selected.sort(key=lambda cluster: cluster.start_frame)
        return selected

    def _remap_frame_token_indices(
        self,
        frame: _FrameKVState,
        old_to_new: torch.Tensor,
    ) -> _FrameKVState:
        token_indices = frame.token_indices.to(device=old_to_new.device, dtype=torch.long)
        remapped = old_to_new.index_select(0, token_indices)
        if torch.any(remapped < 0):
            bad = torch.nonzero(remapped < 0, as_tuple=False).flatten()
            missing = token_indices.index_select(0, bad[:8]).detach().cpu().tolist()
            raise RuntimeError(f"selected frame tokens were evicted unexpectedly: frame={frame.frame_id}, tokens={missing}")
        context_indices = frame.context_indices.to(device=old_to_new.device, dtype=torch.long)
        remapped_context = old_to_new.index_select(0, context_indices)
        if torch.any(remapped_context < 0):
            bad = torch.nonzero(remapped_context < 0, as_tuple=False).flatten()
            missing = context_indices.index_select(0, bad[:8]).detach().cpu().tolist()
            raise RuntimeError(
                f"selected frame context tokens were evicted unexpectedly: frame={frame.frame_id}, tokens={missing}"
            )
        return _FrameKVState(
            frame_id=frame.frame_id,
            sample_index=frame.sample_index,
            timestamp=frame.timestamp,
            token_indices=remapped.detach().clone(),
            context_indices=remapped_context.detach().clone(),
            context_positions=frame.context_positions,
            positions=frame.positions,
            key_vec=frame.key_vec,
            visual_embeds=frame.visual_embeds,
            deepstack_embeds=frame.deepstack_embeds,
            grid_thw=frame.grid_thw,
            image=frame.image,
            source_images=frame.source_images,
            source_sample_indices=frame.source_sample_indices,
            source_timestamps=frame.source_timestamps,
        )

    def _compact_visual_inputs(
        self,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        keep_positions: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        if visual_pos_masks is None:
            return None, None if deepstack_visual_embeds is None else deepstack_visual_embeds

        mask = visual_pos_masks.detach()
        if mask.dim() == 2:
            flat_mask = mask[0]
        else:
            flat_mask = mask.reshape(-1)
        keep_positions = keep_positions.to(device=flat_mask.device, dtype=torch.long)
        compact_flat_mask = flat_mask.index_select(0, keep_positions).view(1, -1)

        if not isinstance(deepstack_visual_embeds, list) or not deepstack_visual_embeds:
            return compact_flat_mask, deepstack_visual_embeds

        visual_positions = torch.nonzero(flat_mask.to(dtype=torch.bool), as_tuple=False).flatten().long()
        if visual_positions.numel() == 0:
            return compact_flat_mask, deepstack_visual_embeds
        rank_map = torch.full((int(flat_mask.numel()),), -1, device=flat_mask.device, dtype=torch.long)
        rank_map[visual_positions] = torch.arange(visual_positions.numel(), device=flat_mask.device)
        kept_visual_positions = keep_positions[flat_mask.index_select(0, keep_positions).to(dtype=torch.bool)]
        kept_ranks = rank_map.index_select(0, kept_visual_positions)
        if torch.any(kept_ranks < 0):
            raise RuntimeError("failed to remap compact DeepStack visual ranks.")

        compact_deepstack: List[torch.Tensor] = []
        for embeds in deepstack_visual_embeds:
            if not isinstance(embeds, torch.Tensor):
                return compact_flat_mask, deepstack_visual_embeds
            layer_embeds = embeds.detach()
            had_batch = layer_embeds.dim() == 3 and int(layer_embeds.shape[0]) == 1
            if had_batch:
                layer_embeds_2d = layer_embeds[0]
            else:
                layer_embeds_2d = layer_embeds
            selected = layer_embeds_2d.index_select(0, kept_ranks.to(layer_embeds_2d.device))
            compact_deepstack.append(selected.unsqueeze(0) if had_batch else selected)
        return compact_flat_mask, compact_deepstack

    def _evict_clustered_long_frame_state(
        self,
        selection: Dict[str, Any],
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        prune_hidden: torch.Tensor,
        input_seq_len: int,
        video_positions: torch.Tensor,
        rope_position_ids: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Dict[str, Any],
    ]:
        short_frames: List[_FrameKVState] = list(selection.get("short") or [])
        long_source_count = len(selection.get("long_source") or [])
        search_candidate_count = len(selection.get("search_candidates") or [])
        if long_source_count <= 0:
            return raw_lower_kv, prune_hidden, visual_pos_masks, deepstack_visual_embeds, selection

        device = video_positions.device
        prefix_old = torch.arange(0, int(video_positions[0].item()), device=device, dtype=torch.long)
        suffix_old = torch.arange(int(video_positions[-1].item()) + 1, input_seq_len, device=device, dtype=torch.long)
        keep_parts = [prefix_old, suffix_old]
        keep_parts.extend(frame.context_indices.to(device=device, dtype=torch.long) for frame in short_frames)
        keep_positions = torch.unique(torch.cat([part for part in keep_parts if part.numel() > 0]), sorted=True)

        old_to_new = torch.full((input_seq_len,), -1, device=device, dtype=torch.long)
        old_to_new[keep_positions] = torch.arange(keep_positions.numel(), device=device, dtype=torch.long)
        compact_prefix = old_to_new.index_select(0, prefix_old) if prefix_old.numel() > 0 else prefix_old
        compact_suffix = old_to_new.index_select(0, suffix_old) if suffix_old.numel() > 0 else suffix_old
        compact_short_frames = [self._remap_frame_token_indices(frame, old_to_new) for frame in short_frames]

        compact_raw_lower_kv: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, entry in raw_lower_kv.items():
            key = entry["k"]
            value = entry["v"]
            gather = keep_positions.to(device=key.device, dtype=torch.long)
            compact_raw_lower_kv[int(layer_idx)] = {
                "k": key.index_select(2, gather).detach(),
                "v": value.index_select(2, gather.to(device=value.device)).detach(),
            }

        compact_hidden = prune_hidden.index_select(1, keep_positions.to(device=prune_hidden.device)).detach()
        compact_visual_mask, compact_deepstack = self._compact_visual_inputs(
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            keep_positions=keep_positions,
        )

        compact_selection = dict(selection)
        compact_selection["short"] = compact_short_frames
        compact_selection["long_source"] = []
        compact_selection["search_candidates"] = []
        compact_selection["long_source_count"] = int(long_source_count)
        compact_selection["short_search_candidate_count"] = int(search_candidate_count)
        compact_selection["evicted_long_frame_count"] = int(long_source_count)
        compact_selection["evicted_raw_token_count"] = int(input_seq_len - keep_positions.numel())
        compact_selection["_compact_prefix_positions"] = compact_prefix.detach().clone()
        compact_selection["_compact_suffix_positions"] = compact_suffix.detach().clone()
        compact_selection["_prefix_rope_positions"] = self._select_rope_positions(rope_position_ids, prefix_old).detach().clone()
        compact_selection["_suffix_rope_positions"] = self._select_rope_positions(rope_position_ids, suffix_old).detach().clone()
        return compact_raw_lower_kv, compact_hidden, compact_visual_mask, compact_deepstack, compact_selection

    def _rope_position_mode(self) -> str:
        mode = str(self.config.get("rope_position_mode", "relative")).strip().lower()
        if mode not in {"relative", "absolute"}:
            raise ValueError(f"Unsupported rope_position_mode={mode!r}; expected 'relative' or 'absolute'.")
        return mode

    def _rotate_half(self, tensor: torch.Tensor) -> torch.Tensor:
        half = tensor.shape[-1] // 2
        return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)

    def _fallback_apply_rope(self, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if cos.dim() == 4 and cos.shape[0] == 3:
            cos = cos[0]
            sin = sin[0]
        while cos.dim() < key.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (key * cos) + (self._rotate_half(key) * sin)

    def _rotary_from_positions(self, rotary_emb, key: torch.Tensor, positions: torch.Tensor):
        positions = positions.to(device=key.device, dtype=torch.long)
        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and getattr(config, "text_config", None) is not None:
            hidden_size = getattr(config.text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = key.shape[1] * key.shape[-1]
        hidden_size = int(hidden_size)
        seq_len = int(positions.shape[-1])
        dummy = key.new_zeros((1, seq_len, hidden_size))
        if positions.dim() == 1:
            pos_for_rotary = positions.view(1, -1)
        elif positions.dim() == 2 and int(positions.shape[0]) == 3:
            pos_for_rotary = positions[:, None, :].contiguous()
        elif positions.dim() == 3:
            pos_for_rotary = positions
        else:
            pos_for_rotary = positions.view(1, -1)
        try:
            return rotary_emb(dummy, pos_for_rotary)
        except Exception:
            pos2 = positions[0].view(1, -1) if positions.dim() > 1 else positions.view(1, -1)
            try:
                return rotary_emb(dummy, pos2)
            except TypeError:
                return rotary_emb(pos2)

    def _apply_rope_to_key(self, raw_key: torch.Tensor, positions: torch.Tensor, rotary_emb) -> torch.Tensor:
        if raw_key.shape[2] == 0:
            return raw_key
        cos, sin = self._rotary_from_positions(rotary_emb, raw_key, positions)
        if _qwen3_apply_rope is not None:
            try:
                _q, key = _qwen3_apply_rope(raw_key, raw_key, cos, sin)
                return key
            except Exception:
                pass
        return self._fallback_apply_rope(raw_key, cos, sin)

    def _compose_selected_positions(
        self,
        prefix_positions: torch.Tensor,
        clusters: Sequence[_LongKVCluster],
        short_frames: Sequence[_FrameKVState],
        suffix_positions: torch.Tensor,
        selected_len: int,
    ) -> torch.Tensor:
        mode = self._rope_position_mode()
        if mode == "relative":
            base = torch.arange(selected_len, device=prefix_positions.device, dtype=torch.long)
            return base.view(1, -1).expand(3, -1).contiguous()

        position_parts: List[torch.Tensor] = [prefix_positions.long()]
        for cluster in clusters:
            position_parts.append(cluster.positions.round().long().to(prefix_positions.device))
        for frame in short_frames:
            position_parts.append(frame.context_positions.long().to(prefix_positions.device))
        position_parts.append(suffix_positions.long())
        positions = torch.cat(position_parts, dim=-1)
        if int(positions.shape[-1]) != selected_len:
            raise RuntimeError(
                f"selected position count {int(positions.shape[-1])} != selected token count {selected_len}."
            )
        return positions
