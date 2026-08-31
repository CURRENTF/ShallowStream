"""Layerwise visual-token voting for OneVision historical frame retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class OneVisionTokenVoteMixin:
    def _mean_token_vote_attention(
        self,
        *,
        q_rot: torch.Tensor,
        k_raw: torch.Tensor,
        key_indices: torch.Tensor,
        key_positions: Optional[torch.Tensor] = None,
        rope_theta: float,
        scaling: float,
    ) -> torch.Tensor:
        """Compute exact mean softmax attention without materializing full QK logits."""

        query_heads = int(q_rot.shape[1])
        key_heads = int(k_raw.shape[1])
        if query_heads % key_heads != 0:
            raise RuntimeError("OneVision token-vote Q/K head mismatch")
        groups = query_heads // key_heads
        q_grouped = q_rot.float().reshape(
            int(q_rot.shape[0]),
            key_heads,
            groups,
            int(q_rot.shape[2]),
            int(q_rot.shape[3]),
        )
        chunk_size = 4096
        log_norm = torch.full(
            q_grouped.shape[:-1],
            -torch.inf,
            device=q_grouped.device,
            dtype=torch.float32,
        )

        def chunk_logits(start: int, end: int) -> torch.Tensor:
            indices = key_indices[start:end]
            selected = k_raw.index_select(2, indices).to(
                device=q_rot.device,
                dtype=q_rot.dtype,
            )
            positions = (
                key_positions.index_select(0, indices.to(key_positions.device))
                if isinstance(key_positions, torch.Tensor)
                else indices
            ).to(device=q_rot.device, dtype=torch.long)
            rotated = self._apply_rope(selected, positions, rope_theta).float()
            return torch.einsum(
                "bhgqd,bhkd->bhgqk",
                q_grouped,
                rotated,
            ).mul_(scaling)

        key_count = int(key_indices.numel())
        for start in range(0, key_count, chunk_size):
            logits = chunk_logits(start, min(start + chunk_size, key_count))
            log_norm = torch.logaddexp(log_norm, torch.logsumexp(logits, dim=-1))

        attention_chunks = []
        for start in range(0, key_count, chunk_size):
            logits = chunk_logits(start, min(start + chunk_size, key_count))
            probabilities = torch.exp(logits - log_norm.unsqueeze(-1))
            attention_chunks.append(probabilities.mean(dim=(0, 1, 2, 3)))
        return torch.cat(attention_chunks, dim=0)

    def _retrieval_score_strategy(self) -> str:
        strategy = str(
            self.config.get("retrieval_score_strategy", "final_layer_cosine")
            or "final_layer_cosine"
        ).strip().lower()
        aliases = {
            "cosine": "final_layer_cosine",
            "final_layer_cosine": "final_layer_cosine",
            "layer_token_vote": "shallow_layer_token_vote",
            "shallow_layer_token_vote": "shallow_layer_token_vote",
        }
        if strategy not in aliases:
            raise ValueError(f"Unsupported retrieval_score_strategy={strategy!r}")
        return aliases[strategy]

    def _select_diverse_vote_frames(
        self,
        *,
        ranked_frames: Sequence[int],
        frame_vecs: torch.Tensor,
        frame_budget: int,
    ) -> List[int]:
        pool = [int(frame) for frame in ranked_frames]
        target = min(max(int(frame_budget), 0), len(pool))
        if target <= 1:
            return pool[:target]
        representations = torch.nn.functional.normalize(
            frame_vecs.index_select(
                0,
                torch.tensor(pool, device=frame_vecs.device, dtype=torch.long),
            ).detach().float().cpu(),
            dim=-1,
        )
        distances = 1.0 - representations @ representations.transpose(0, 1)
        distances.fill_diagonal_(-1.0)
        pairs = [
            (float(distances[left, right].item()), -(left + right), -left, left, right)
            for left in range(len(pool))
            for right in range(left + 1, len(pool))
        ]
        _distance, _rank_sum, _left_rank, left, right = max(pairs)
        selected = [left, right]
        remaining = set(range(len(pool))) - set(selected)
        while len(selected) < target:
            best = max(
                remaining,
                key=lambda candidate: (
                    float(distances[candidate, selected].min().item()),
                    -candidate,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return [pool[index] for index in selected]

    def _select_shallow_layer_token_vote_frames(
        self,
        *,
        candidate_frames: Sequence[int],
        frame_vecs: torch.Tensor,
        frame_budget: int,
        device: str,
    ) -> Tuple[List[int], Dict[str, Any]]:
        candidates = [int(frame) for frame in candidate_frames]
        if not candidates or frame_budget <= 0:
            return [], {
                "retrieval_score_strategy": "shallow_layer_token_vote",
                "retrieval_vote_selected_frame_ids": [],
            }
        query_layers = self.state.get("last_retrieval_query_q")
        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if not isinstance(query_layers, dict) or not isinstance(lower_kv, dict):
            raise RuntimeError("OneVision token voting requires captured shallow Q/K")
        if not isinstance(frame_spans, list):
            raise RuntimeError("OneVision token voting requires frame spans")

        layers, _, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        layer_start = max(int(self.config.get("retrieval_vote_layer_start", 0)), 0)
        if layer_start >= prune:
            raise ValueError("retrieval_vote_layer_start must be below prune_layer")
        layer_indices = list(range(layer_start, prune))
        query_mode = str(
            self.config.get("retrieval_vote_query_token_mode", "all_mean")
            or "all_mean"
        ).strip().lower()
        if query_mode not in {"all_mean", "prompt_last"}:
            raise ValueError(f"Unsupported retrieval_vote_query_token_mode={query_mode!r}")

        visual_positions: List[int] = []
        token_frame_offsets: List[int] = []
        for offset, frame_id in enumerate(candidates):
            if frame_id < 0 or frame_id >= len(frame_spans):
                raise IndexError(f"Token-vote frame id is out of range: {frame_id}")
            span = frame_spans[frame_id]
            start = int(span.get("visual_start", -1))
            end = int(span.get("visual_end", -1))
            if start < 0 or end <= start:
                raise RuntimeError(f"Invalid visual span for frame {frame_id}")
            visual_positions.extend(range(start, end))
            token_frame_offsets.extend([offset] * (end - start))
        frame_offsets = torch.tensor(token_frame_offsets, dtype=torch.long)
        frame_votes = torch.zeros(len(candidates), dtype=torch.long)
        frame_attention = torch.zeros(len(candidates), dtype=torch.float64)
        per_layer_selected: Dict[str, List[int]] = {}
        token_topk = max(
            int(self.config.get("retrieval_vote_topk_tokens_per_layer", 64)), 1
        )
        effective_topk = min(token_topk, len(visual_positions))
        rope_theta = self._rope_base()

        for layer_idx in layer_indices:
            q_entry = query_layers.get(layer_idx)
            k_entry = lower_kv.get(layer_idx)
            if not isinstance(q_entry, dict) or not isinstance(k_entry, dict):
                raise RuntimeError(f"Missing token-vote Q/K at layer {layer_idx}")
            q_raw = q_entry.get("q")
            q_positions = q_entry.get("positions")
            k_raw = k_entry.get("k")
            if not all(isinstance(value, torch.Tensor) for value in (q_raw, q_positions, k_raw)):
                raise RuntimeError(f"Invalid token-vote Q/K at layer {layer_idx}")
            query_indices = (
                torch.tensor([int(q_raw.shape[2]) - 1], device=q_raw.device)
                if query_mode == "prompt_last"
                else torch.arange(int(q_raw.shape[2]), device=q_raw.device)
            )
            key_indices = torch.tensor(
                visual_positions, device=k_raw.device, dtype=torch.long
            )
            key_positions_value = k_entry.get("pos")
            if isinstance(key_positions_value, torch.Tensor):
                key_positions = key_positions_value.to(
                    device=k_raw.device,
                    dtype=torch.long,
                ).reshape(-1)
                if int(key_positions.numel()) != int(k_raw.shape[2]):
                    raise RuntimeError(
                        "Invalid token-vote lower-cache positions at layer "
                        f"{layer_idx}: got {int(key_positions.numel())}, "
                        f"expected {int(k_raw.shape[2])}"
                    )
            else:
                key_positions = None
            q_selected = q_raw.index_select(2, query_indices)
            q_pos = q_positions.index_select(0, query_indices).to(q_selected.device)
            q_rot = self._apply_rope(q_selected, q_pos, rope_theta)
            attention = layers[layer_idx].self_attn
            scaling = float(getattr(attention, "scaling", q_rot.shape[-1] ** -0.5))
            token_attention = self._mean_token_vote_attention(
                q_rot=q_rot,
                k_raw=k_raw,
                key_indices=key_indices,
                key_positions=key_positions,
                rope_theta=rope_theta,
                scaling=scaling,
            )
            selected_tokens = torch.topk(
                token_attention, k=effective_topk, largest=True, sorted=True
            ).indices
            selected_offsets = frame_offsets.index_select(
                0, selected_tokens.detach().cpu()
            )
            selected_attention = token_attention.index_select(
                0, selected_tokens
            ).detach().double().cpu()
            frame_votes.scatter_add_(
                0, selected_offsets, torch.ones_like(selected_offsets)
            )
            frame_attention.scatter_add_(0, selected_offsets, selected_attention)
            per_layer_selected[str(layer_idx)] = [
                candidates[offset] for offset in selected_offsets.tolist()
            ]

        ranked_offsets = sorted(
            range(len(candidates)),
            key=lambda offset: (
                -int(frame_votes[offset].item()),
                -float(frame_attention[offset].item()),
                candidates[offset],
            ),
        )
        ranked_frames = [candidates[offset] for offset in ranked_offsets]
        diversity_mode = str(
            self.config.get("retrieval_vote_diversity_mode", "off") or "off"
        ).strip().lower()
        if diversity_mode not in {"off", "divprune_maxmin"}:
            raise ValueError(f"Unsupported retrieval_vote_diversity_mode={diversity_mode!r}")
        pool_multiplier = max(
            int(self.config.get("retrieval_vote_diversity_pool_multiplier", 4)), 1
        )
        pool_size = min(len(ranked_frames), int(frame_budget) * pool_multiplier)
        if diversity_mode == "divprune_maxmin":
            selected = self._select_diverse_vote_frames(
                ranked_frames=ranked_frames[:pool_size],
                frame_vecs=frame_vecs,
                frame_budget=frame_budget,
            )
        else:
            selected = ranked_frames[: min(int(frame_budget), len(ranked_frames))]
        stats = {
            "retrieval_score_strategy": "shallow_layer_token_vote",
            "retrieval_vote_layer_indices": layer_indices,
            "retrieval_vote_topk_tokens_per_layer": effective_topk,
            "retrieval_vote_query_token_mode": query_mode,
            "retrieval_vote_selected_frame_ids": selected,
            "retrieval_vote_diversity_mode": diversity_mode,
            "retrieval_vote_diversity_pool_size": pool_size,
            "retrieval_vote_frame_scores": [
                {
                    "frame_id": candidates[offset],
                    "votes": int(frame_votes[offset].item()),
                    "attention_tiebreak": float(frame_attention[offset].item()),
                }
                for offset in ranked_offsets
            ],
            "retrieval_vote_layer_selected_frame_ids": per_layer_selected,
        }
        return selected, stats
