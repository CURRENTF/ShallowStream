"""Layerwise history-attention gate for the OneVision runtime."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from src.shallowstream.history_decay_gate import score_history_attention_layers


class OneVisionHistoryDecayMixin:
    def _history_decay_query_indices(
        self,
        generation_prompt: str,
        raw_question: str,
        expected_tokens: int,
    ) -> List[int]:
        question = str(raw_question or "").strip()
        if not question:
            raise ValueError("history_layer_decay requires non-empty question text")
        start = str(generation_prompt).find(question)
        if start < 0:
            raise RuntimeError(
                "history_layer_decay could not locate the raw question in the generation prompt"
            )
        encoded = self.tokenizer(
            generation_prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.get("offset_mapping")
        if isinstance(offsets, torch.Tensor):
            offsets = offsets[0].detach().cpu().tolist()
        if not isinstance(offsets, list) or len(offsets) != int(expected_tokens):
            raise RuntimeError(
                "history_layer_decay tokenizer offsets do not match prompt tokens"
            )
        end = start + len(question)
        indices = [
            index
            for index, pair in enumerate(offsets)
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and int(pair[1]) > start
            and int(pair[0]) < end
            and int(pair[1]) > int(pair[0])
        ]
        if not indices:
            raise RuntimeError("history_layer_decay found no question-content tokens")
        return indices

    def _build_history_decay_observation(self, raw_question: str) -> Dict[int, Dict[str, float]]:
        capture = self.state.pop("last_history_decay_query", None)
        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if not isinstance(capture, dict) or not isinstance(lower_kv, dict):
            raise RuntimeError("history_layer_decay is missing shallow Q/K capture")
        if not isinstance(frame_spans, list) or not frame_spans:
            raise RuntimeError("history_layer_decay is missing visual frame spans")

        query_by_layer = capture.get("layers")
        prompt_ids = capture.get("prompt_ids")
        generation_prompt = str(capture.get("generation_prompt", ""))
        if not isinstance(query_by_layer, dict) or not isinstance(prompt_ids, torch.Tensor):
            raise RuntimeError("history_layer_decay has incomplete question capture")
        query_indices = self._history_decay_query_indices(
            generation_prompt,
            raw_question,
            int(prompt_ids.shape[1]),
        )

        recent_frames = min(
            max(0, int(self.config.get("retrieval_recent_frames", 0))),
            len(frame_spans),
        )
        history_frames = len(frame_spans) - recent_frames
        visual_ranges = [
            torch.arange(int(span["visual_start"]), int(span["visual_end"]), dtype=torch.long)
            for span in frame_spans
        ]
        history_token_count = sum(int(value.numel()) for value in visual_ranges[:history_frames])
        recent_token_count = sum(int(value.numel()) for value in visual_ranges[history_frames:])
        if history_frames <= 0 or recent_frames <= 0:
            return {
                int(layer): {
                    "history_enrichment": 0.0,
                    "history_attention_mass": 0.0,
                    "uniform_history_mass": 0.0,
                    "head_history_consensus": 0.0,
                    "query_history_consensus": 0.0,
                    "head_enrichment_std": 0.0,
                    "query_enrichment_std": 0.0,
                    "history_frame_count": float(history_frames),
                    "recent_frame_count": float(recent_frames),
                    "history_token_count": float(history_token_count),
                    "recent_token_count": float(recent_token_count),
                    "attention_available": 0.0,
                }
                for layer in sorted(query_by_layer)
            }

        visual_indices = torch.cat(visual_ranges)
        query_index = torch.tensor(query_indices, dtype=torch.long)
        layers, _embed_tokens, _norm, _lm_head = self._get_lm_components()
        observations: Dict[int, Dict[str, float]] = {}
        for layer_idx in sorted(query_by_layer):
            query_entry = query_by_layer[layer_idx]
            key_entry = lower_kv.get(layer_idx)
            if not isinstance(query_entry, dict) or not isinstance(key_entry, dict):
                raise RuntimeError(
                    f"history_layer_decay is missing Q/K capture at layer {layer_idx}"
                )
            q_raw = query_entry.get("q")
            q_positions = query_entry.get("positions")
            k_raw = key_entry.get("k")
            if not all(isinstance(value, torch.Tensor) for value in (q_raw, q_positions, k_raw)):
                raise RuntimeError(
                    f"history_layer_decay has invalid Q/K capture at layer {layer_idx}"
                )
            target_device = q_raw.device
            q_idx = query_index.to(device=target_device)
            k_idx = visual_indices.to(device=k_raw.device)
            q_selected = q_raw.index_select(2, q_idx)
            k_selected = k_raw.index_select(2, k_idx).to(
                device=target_device,
                dtype=q_selected.dtype,
            )
            q_pos = q_positions.index_select(0, q_idx.to(device=q_positions.device)).to(
                device=target_device,
                dtype=torch.long,
            )
            key_positions_value = key_entry.get("pos")
            if isinstance(key_positions_value, torch.Tensor):
                key_positions = key_positions_value.to(
                    device=k_raw.device,
                    dtype=torch.long,
                ).reshape(-1)
                if int(key_positions.numel()) != int(k_raw.shape[2]):
                    raise RuntimeError(
                        "Invalid history-decay lower-cache positions at layer "
                        f"{layer_idx}: got {int(key_positions.numel())}, "
                        f"expected {int(k_raw.shape[2])}"
                    )
                k_pos = key_positions.index_select(0, k_idx).to(device=target_device)
            else:
                k_pos = k_idx.to(device=target_device, dtype=torch.long)
            rope_theta = self._rope_base()
            q_rot = self._apply_rope(q_selected, q_pos, rope_theta)
            k_rot = self._apply_rope(k_selected, k_pos, rope_theta)
            if int(q_rot.shape[1]) != int(k_rot.shape[1]):
                if int(q_rot.shape[1]) % int(k_rot.shape[1]) != 0:
                    raise RuntimeError(
                        "history_layer_decay Q/KV head counts are incompatible: "
                        f"q={int(q_rot.shape[1])}, kv={int(k_rot.shape[1])}"
                    )
                k_rot = self._repeat_kv(
                    k_rot,
                    int(q_rot.shape[1]) // int(k_rot.shape[1]),
                )
            attention = layers[int(layer_idx)].self_attn
            scaling = float(getattr(attention, "scaling", q_rot.shape[-1] ** -0.5))
            logits = torch.matmul(
                q_rot.float(),
                k_rot.float().transpose(2, 3),
            ).mul_(scaling)
            history_lse = torch.logsumexp(
                logits[..., :history_token_count], dim=-1
            )
            recent_lse = torch.logsumexp(
                logits[..., history_token_count:], dim=-1
            )
            raw_log_odds = history_lse - recent_lse
            enrichment = raw_log_odds - math.log(
                history_token_count / float(recent_token_count)
            )
            head_scores = enrichment.mean(dim=2)
            query_scores = enrichment.mean(dim=1)
            mass = torch.sigmoid(raw_log_odds)
            observations[int(layer_idx)] = {
                "history_enrichment": float(enrichment.mean().item()),
                "history_attention_mass": float(mass.mean().item()),
                "uniform_history_mass": float(
                    history_token_count / (history_token_count + recent_token_count)
                ),
                "head_history_consensus": float(
                    (head_scores > 0).float().mean().item()
                ),
                "query_history_consensus": float(
                    (query_scores > 0).float().mean().item()
                ),
                "head_enrichment_std": float(
                    head_scores.std(unbiased=False).item()
                ),
                "query_enrichment_std": float(
                    query_scores.std(unbiased=False).item()
                ),
                "head_history_mass_std": float(
                    mass.mean(dim=2).std(unbiased=False).item()
                ),
                "query_history_mass_std": float(
                    mass.mean(dim=1).std(unbiased=False).item()
                ),
                "history_frame_count": float(history_frames),
                "recent_frame_count": float(recent_frames),
                "history_token_count": float(history_token_count),
                "recent_token_count": float(recent_token_count),
                "attention_available": 1.0,
            }
        return observations

    def _score_history_layer_decay_gate(self, source: str = "frame_retrieval") -> Dict[str, Any]:
        threshold = float(
            self.config.get("task_gate_history_decay_threshold", 0.0)
        )
        observations = self.state.get("last_history_decay_observation")
        if not isinstance(observations, dict):
            raise RuntimeError("history_layer_decay is missing layerwise observations")
        available = {
            int(layer): float(features["history_attention_mass"])
            for layer, features in observations.items()
            if isinstance(features, dict)
            and float(features.get("attention_available", 0.0)) > 0.0
        }
        if len(available) < 2:
            decision: Dict[str, Any] = {
                "enabled": True,
                "mode": "history_layer_decay",
                "source": source,
                "predicted_task_type": "realtime",
                "selected_policy": str(
                    self.config.get("task_gate_realtime_policy", "recent_only")
                ),
                "retrieval_enabled": False,
                "score": None,
                "threshold": threshold,
                "reason": "fewer_than_two_layers_with_history",
            }
        else:
            scored = score_history_attention_layers(
                available,
                str(self.config.get("task_gate_history_decay_variant", "endpoint_delta")),
                threshold,
            )
            retrieval_enabled = bool(float(scored["score"]) >= threshold)
            decision = {
                "enabled": True,
                "mode": "history_layer_decay",
                "source": source,
                "predicted_task_type": "backward" if retrieval_enabled else "realtime",
                "selected_policy": (
                    "retrieval"
                    if retrieval_enabled
                    else str(self.config.get("task_gate_realtime_policy", "recent_only"))
                ),
                "retrieval_enabled": retrieval_enabled,
                **scored,
            }
        decision.update(
            {
                "query_source": "question_text_content_q_to_video_k",
                "representation": "matching_head_scaled_qk_history_attention_across_layers",
                "rule": "enable_retrieval_if_layerwise_log_history_attention_change_gte_threshold",
                "observation_features": {
                    str(layer): {
                        name: float(value)
                        for name, value in sorted(features.items())
                    }
                    for layer, features in sorted(observations.items())
                    if isinstance(features, dict)
                },
            }
        )
        return decision
