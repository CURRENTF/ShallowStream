"""Qwen3VLTaskGateMixin implementation."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.shallowstream.common import select_temporal_retrieval_ids
from src.shallowstream.evidence_retrieval import (
    evidence_retrieval_backend,
    get_siglip_evidence_retriever,
    resolve_evidence_query,
)
from src.shallowstream.history_decay_gate import score_history_attention_layers
from src.shallowstream.task_gate import (
    TASK_GATE_INPUT_SOURCES,
    build_anchor_hidden_decision,
    build_query_choice_decision,
    build_query_choice_router_prompt,
    normalize_query_choice_prompt_version,
    query_choice_token_strings,
    query_source_label,
    resolve_query_choice_token_ids,
    resolve_task_gate_input,
    task_gate_input_metadata,
)

from .config import (
    MODEL_NAME,
    _as_bool,
    _as_float,
    _as_int,
    _retrieval_expansion_strategy,
)


class Qwen3VLTaskGateMixin:

    def _evidence_retrieval_backend(self) -> str:
        return evidence_retrieval_backend(self.config)

    def _rerank_siglip_evidence(
        self,
        selection: Dict[str, Any],
        *,
        prompt: str,
    ) -> Dict[str, Any]:
        candidates = list(selection.get("search_candidates") or [])
        recent = list(selection.get("recent") or [])
        question_text = str(
            getattr(self.owner, "_task_gate_query_text", "") or ""
        )
        query, query_source = resolve_evidence_query(
            self.config,
            full_prompt=prompt,
            question_text=question_text,
        )
        images = []
        for frame in candidates:
            image = getattr(frame, "image", None)
            if image is None:
                raise RuntimeError(
                    "Qwen3-VL SigLIP evidence retrieval requires a retained image for "
                    f"visual unit {getattr(frame, 'frame_id', '?')}"
                )
            images.append(image)
        result = get_siglip_evidence_retriever(self).score(query, images)
        if len(result.scores) != len(candidates):
            raise RuntimeError("Qwen3-VL SigLIP score count does not match candidate units")

        scored = sorted(
            zip(result.scores, candidates),
            key=lambda item: (-float(item[0]), int(item[1].frame_id)),
        )
        topk = min(self._retrieval_topk_units(), len(scored))
        retrieved_seed = [frame for _score, frame in scored[:topk]]
        candidate_by_id = {
            int(frame.frame_id): frame for frame in [*candidates, *recent]
        }
        selected_ids, reference_expanded_ids = select_temporal_retrieval_ids(
            (int(frame.frame_id) for frame in retrieved_seed),
            (int(frame.frame_id) for _score, frame in scored),
            candidate_by_id,
            previous=self._retrieval_expand_prev_units(),
            following=self._retrieval_expand_next_units(),
            previous_stride=self._retrieval_expand_prev_stride_units(),
            following_stride=self._retrieval_expand_next_stride_units(),
            strategy=_retrieval_expansion_strategy(self.config),
        )
        retrieved = [candidate_by_id[frame_id] for frame_id in selected_ids]
        selected_short = sorted(
            {int(frame.frame_id): frame for frame in [*retrieved, *recent]}.values(),
            key=lambda frame: int(frame.frame_id),
        )
        score_by_id = {
            int(frame.frame_id): float(score) for score, frame in scored
        }
        metadata = dict(result.metadata)
        metadata.update(
            {
                "query_source": query_source,
                "candidate_unit_ids": [int(frame.frame_id) for frame in candidates],
                "candidate_scores": [
                    float(score_by_id[int(frame.frame_id)]) for frame in candidates
                ],
                "retrieved_seed_unit_ids": [
                    int(frame.frame_id) for frame in retrieved_seed
                ],
                "retrieved_expanded_unit_ids": [
                    int(frame.frame_id) for frame in retrieved
                ],
                "retrieval_reference_expanded_unit_ids": reference_expanded_ids,
            }
        )
        reranked = dict(selection)
        reranked.update(
            {
                "short": selected_short,
                "clusters": [],
                "retrieved": retrieved,
                "retrieved_seed": retrieved_seed,
                "retrieval_reference_expanded_unit_ids": reference_expanded_ids,
                "short_scores": [
                    (float(score), frame) for score, frame in scored
                ],
                "evidence_retrieval": metadata,
            }
        )
        stats = dict(selection.get("token_selection_stats") or {})
        stats.update(
            {
                "retrieval_selection_granularity": "unit",
                "evidence_retrieval_backend": "siglip",
                "evidence_retrieval_query_source": query_source,
                "evidence_retrieval_model_path": metadata["model_path"],
                "evidence_retrieval_candidate_count": len(candidates),
                "evidence_retrieval_seed_units": metadata[
                    "retrieved_seed_unit_ids"
                ],
            }
        )
        reranked["token_selection_stats"] = stats
        return reranked

    def _task_gate_mode(self) -> str:
        mode = str(self.config.get("task_gate_mode", "off") or "off").strip().lower()
        supported = {
            "off",
            "attention_distribution",
            "history_layer_decay",
            "anchor_hidden",
            "anchor_kq",
            "decision_replay",
            "latest_unit_score",
            "query_text_nb",
            "query_choice_logits",
            "recent_context_sufficiency",
            "query_semantic_zeroshot",
            "query_heuristic_zeroshot",
            "attention_skew",
            "attention_probe",
        }
        if mode not in supported:
            raise ValueError(f"Unsupported task_gate_mode={mode!r}; expected one of {sorted(supported)}")
        input_source = str(
            self.config.get("task_gate_input_source", "full_prompt")
            or "full_prompt"
        ).strip().lower()
        if input_source not in TASK_GATE_INPUT_SOURCES:
            raise ValueError(
                f"Unsupported task_gate_input_source={input_source!r}; expected one of "
                f"{list(TASK_GATE_INPUT_SOURCES)!r}"
            )
        shared_input_modes = {
            "anchor_hidden",
            "anchor_kq",
            "history_layer_decay",
            "query_choice_logits",
        }
        if mode not in shared_input_modes and input_source != "full_prompt":
            raise ValueError(
                "task_gate_input_source=question_text is supported only for "
                f"{sorted(shared_input_modes)!r}; got task_gate_mode={mode!r}"
            )
        if mode == "latest_unit_score":
            if int(self.config.get("retrieval_recent_units", 0) or 0) <= 0:
                raise ValueError(
                    "task_gate_mode=latest_unit_score requires retrieval_recent_units > 0"
                )
            threshold = float(
                self.config.get("task_gate_latest_unit_score_threshold", 0.0095)
            )
            if not math.isfinite(threshold):
                raise ValueError(
                    "task_gate_latest_unit_score_threshold must be finite"
                )
        if mode == "recent_context_sufficiency":
            recent_units = int(
                self.config.get("task_gate_recent_sufficiency_units", 2) or 0
            )
            threshold = float(
                self.config.get("task_gate_recent_sufficiency_threshold", 0.0)
            )
            if recent_units <= 0:
                raise ValueError(
                    "task_gate_recent_sufficiency_units must be positive"
                )
            if not math.isfinite(threshold):
                raise ValueError(
                    "task_gate_recent_sufficiency_threshold must be finite"
                )
        return mode

    def _load_task_gate_replay_decisions(self) -> Optional[Dict[str, bool]]:
        if self._task_gate_mode() != "decision_replay":
            return None
        raw_path = str(self.config.get("task_gate_replay_path", "") or "").strip()
        if not raw_path:
            raise ValueError("task_gate_mode=decision_replay requires task_gate_replay_path")
        replay_path = self._resolve_repo_path(raw_path)
        try:
            with open(replay_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise FileNotFoundError(f"Failed to read task gate replay decisions: {replay_path}") from exc
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Task gate replay must be a non-empty JSON object: {replay_path}")
        decisions: Dict[str, bool] = {}
        for sample_id, value in payload.items():
            if not isinstance(value, bool):
                raise ValueError(
                    f"Task gate replay decision for sample {sample_id!r} must be boolean"
                )
            decisions[str(sample_id)] = value
        return decisions

    def _score_task_gate_replay(self, source: str) -> Dict[str, Any]:
        sample_id = str(getattr(self.owner, "_task_gate_sample_id", "") or "").strip()
        if not sample_id:
            raise RuntimeError("decision_replay requires the current sample id")
        decisions = self._task_gate_replay_decisions
        if not isinstance(decisions, dict) or sample_id not in decisions:
            raise KeyError(f"No task gate replay decision for sample id {sample_id!r}")
        retrieval_enabled = bool(decisions[sample_id])
        return {
            "enabled": True,
            "mode": "decision_replay",
            "source": source,
            "sample_id": sample_id,
            "predicted_task_type": "backward" if retrieval_enabled else "realtime",
            "selected_policy": "retrieval"
            if retrieval_enabled
            else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "retrieval_enabled": retrieval_enabled,
            "query_source": "per_sample_decision_replay",
        }

    def _score_latest_unit_gate(
        self,
        selection: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        recent_frames = list(selection.get("recent", []))
        if not recent_frames:
            raise RuntimeError("latest_unit_score gate requires at least one recent unit")
        query_vec = selection.get("query_vec")
        latest_frame = recent_frames[-1]
        latest_key = getattr(latest_frame, "key_vec", None)
        if not isinstance(query_vec, torch.Tensor) or not isinstance(
            latest_key, torch.Tensor
        ):
            raise RuntimeError(
                "latest_unit_score gate requires query_vec and latest unit key_vec"
            )

        score = float(torch.dot(latest_key.float(), query_vec.float()).item())
        if not math.isfinite(score):
            raise RuntimeError("latest_unit_score gate produced a non-finite score")
        threshold = float(
            self.config.get("task_gate_latest_unit_score_threshold", 0.0095)
        )
        retrieval_enabled = bool(score <= threshold)
        return {
            "enabled": True,
            "mode": "latest_unit_score",
            "source": source,
            "predicted_task_type": "backward" if retrieval_enabled else "realtime",
            "selected_policy": (
                "retrieval"
                if retrieval_enabled
                else str(self.config.get("task_gate_realtime_policy", "recent_only"))
            ),
            "retrieval_enabled": retrieval_enabled,
            "score": score,
            "threshold": threshold,
            "metric": "shallow_qk_cosine",
            "representation": (
                "normalized_latest_temporal_unit_k_dot_normalized_question_q"
            ),
            "query_source": "retrieval_query_vector",
            "latest_unit_frame_id": int(getattr(latest_frame, "frame_id")),
            "latest_unit_sample_index": int(getattr(latest_frame, "sample_index")),
            "latest_unit_timestamp": float(getattr(latest_frame, "timestamp")),
            "rule": "enable_retrieval_if_latest_unit_score_lte_threshold",
        }

    def _task_gate_probe_strategy(self) -> str:
        strategy = str(self.config.get("task_gate_probe_strategy", "temporal_topk") or "temporal_topk").strip().lower()
        if strategy in {"temporal", "temporal_topk", "topk"}:
            return "temporal_topk"
        if strategy in {"prototype", "prototype_contrast", "global_local", "contrast"}:
            return "prototype_contrast"
        raise ValueError(f"Unsupported task_gate_probe_strategy={strategy!r}")

    def _task_gate_attention_layer(self, layer_count: int) -> int:
        configured = self.config.get("task_gate_attention_layer")
        layer = (
            _as_int(self.config, "prune_layer") - 1
            if configured is None or str(configured).strip() == ""
            else int(configured)
        )
        shallow_depth = _as_int(self.config, "prune_layer")
        if not 0 <= layer < min(int(layer_count), shallow_depth):
            raise ValueError(
                "task_gate_attention_layer must select a retained shallow layer in "
                f"[0, {min(int(layer_count), shallow_depth) - 1}], got {layer}"
            )
        return layer

    def _task_gate_attention_observation_layers(self, layer_count: int) -> List[int]:
        selected = self._task_gate_attention_layer(layer_count)
        configured = self.config.get("task_gate_attention_observation_layers") or []
        if not isinstance(configured, (list, tuple)):
            raise ValueError("task_gate_attention_observation_layers must be a list")
        layers = sorted({selected, *(int(value) for value in configured)})
        shallow_depth = min(int(layer_count), _as_int(self.config, "prune_layer"))
        invalid = [layer for layer in layers if not 0 <= layer < shallow_depth]
        if invalid:
            raise ValueError(
                "task_gate_attention_observation_layers must contain retained shallow "
                f"layers in [0, {shallow_depth - 1}], got {invalid}"
            )
        return layers

    def _attention_history_enrichment(
        self,
        logits: torch.Tensor,
        *,
        history_token_count: int,
        recent_token_count: int,
    ) -> Dict[str, float]:
        """Summarize matching-head QK logits without a history-size bias.

        ``logits`` is ``[head, query_token, visual_token]`` with history tokens
        first and recent tokens last. The raw log odds equal the attention-mass
        log ratio that a softmax over visual keys would produce. Subtracting
        ``log(history_count / recent_count)`` makes identical history/recent
        logits score exactly zero regardless of region cardinality.
        """

        if logits.dim() != 3:
            raise ValueError(
                "attention-distribution logits must have shape "
                f"[head, query, visual], got {tuple(logits.shape)}"
            )
        history_count = int(history_token_count)
        recent_count = int(recent_token_count)
        if history_count <= 0 or recent_count <= 0:
            raise ValueError("attention-distribution gate requires non-empty history and recent regions")
        if history_count + recent_count != int(logits.shape[-1]):
            raise ValueError(
                "attention-distribution token partition does not match logits: "
                f"history={history_count}, recent={recent_count}, visual={int(logits.shape[-1])}"
            )

        values = logits.float()
        history_lse = torch.logsumexp(values[..., :history_count], dim=-1)
        recent_lse = torch.logsumexp(values[..., history_count:], dim=-1)
        raw_log_odds = history_lse - recent_lse
        cardinality_log_odds = math.log(history_count / float(recent_count))
        enrichment = raw_log_odds - cardinality_log_odds
        head_scores = enrichment.mean(dim=1)
        query_scores = enrichment.mean(dim=0)
        raw_mass = torch.sigmoid(raw_log_odds)
        return {
            "history_enrichment": float(enrichment.mean().item()),
            "history_attention_mass": float(raw_mass.mean().item()),
            "uniform_history_mass": float(history_count / (history_count + recent_count)),
            "head_history_consensus": float((head_scores > 0).float().mean().item()),
            "query_history_consensus": float((query_scores > 0).float().mean().item()),
            "head_enrichment_std": float(head_scores.std(unbiased=False).item()),
            "query_enrichment_std": float(query_scores.std(unbiased=False).item()),
        }

    def _attention_distribution_features_from_raw(
        self,
        *,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        query_indices: Sequence[int],
        unit_indices: Sequence[torch.Tensor],
        recent_unit_count: int,
        rotary_emb,
        attention_module,
    ) -> Dict[str, float]:
        if q_raw.dim() != 4 or k_raw.dim() != 4:
            raise ValueError("attention-distribution Q and K must be rank-4 tensors")
        if int(q_raw.shape[0]) != 1 or int(k_raw.shape[0]) != 1:
            raise ValueError("attention-distribution gate currently requires batch size 1")
        indices = [int(index) for index in query_indices]
        if not indices:
            raise ValueError("attention-distribution gate requires question-content query tokens")
        units = [unit.detach().long().view(-1) for unit in unit_indices if int(unit.numel()) > 0]
        recent_units = min(max(int(recent_unit_count), 0), len(units))
        history_units = len(units) - recent_units
        if history_units <= 0 or recent_units <= 0:
            return {
                "history_enrichment": 0.0,
                "history_attention_mass": 0.0,
                "uniform_history_mass": 0.0,
                "head_history_consensus": 0.0,
                "query_history_consensus": 0.0,
                "head_enrichment_std": 0.0,
                "query_enrichment_std": 0.0,
                "history_unit_count": float(history_units),
                "recent_unit_count": float(recent_units),
                "history_token_count": 0.0,
                "recent_token_count": float(sum(int(unit.numel()) for unit in units[-recent_units:])),
                "attention_available": 0.0,
            }

        query_index = torch.tensor(indices, device=q_raw.device, dtype=torch.long)
        visual_index = torch.cat(units, dim=0).to(device=k_raw.device, dtype=torch.long)
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
        k_rope_positions = k_rope_positions.to(device=target_device, dtype=torch.long)
        q_rope_positions = q_rope_positions.to(device=target_device, dtype=torch.long)
        q_rot = self._apply_rope_to_key(q_selected, q_rope_positions, rotary_emb)
        k_rot = self._apply_rope_to_key(k_selected, k_rope_positions, rotary_emb)
        if int(q_rot.shape[1]) != int(k_rot.shape[1]):
            if int(q_rot.shape[1]) % int(k_rot.shape[1]) != 0:
                raise RuntimeError(
                    "Q/KV head mismatch for attention-distribution gate: "
                    f"q={int(q_rot.shape[1])}, kv={int(k_rot.shape[1])}"
                )
            k_rot = self._repeat_kv(
                k_rot,
                int(q_rot.shape[1]) // int(k_rot.shape[1]),
            )
        scaling = float(
            getattr(attention_module, "scaling", q_rot.shape[-1] ** -0.5)
        )
        logits = torch.matmul(
            q_rot.float(),
            k_rot.float().transpose(2, 3),
        ).mul_(scaling)[0]
        history_token_count = sum(int(unit.numel()) for unit in units[:history_units])
        recent_token_count = sum(int(unit.numel()) for unit in units[history_units:])
        features = self._attention_history_enrichment(
            logits,
            history_token_count=history_token_count,
            recent_token_count=recent_token_count,
        )
        features.update(
            {
                "history_unit_count": float(history_units),
                "recent_unit_count": float(recent_units),
                "history_token_count": float(history_token_count),
                "recent_token_count": float(recent_token_count),
                "attention_available": 1.0,
            }
        )
        return features

    def _attention_distribution_query_indices(
        self,
        text_inputs: Dict[str, Any],
        prompt: str,
    ) -> List[int]:
        question_text = str(
            getattr(self.owner, "_task_gate_query_text", "") or ""
        ).strip()
        prompt_text = str(prompt).strip()
        if not question_text:
            return self._anchor_hidden_content_token_indices(
                text_inputs,
                prompt_text,
            )
        try:
            return self._anchor_hidden_content_token_indices(
                text_inputs,
                question_text,
            )
        except RuntimeError as direct_error:
            # A BPE token at the question/options boundary can differ from the
            # same question tokenized in isolation. Locate the complete prompt
            # in the chat template, then use fast-tokenizer character offsets
            # to select only tokens overlapping the raw question.
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None or not prompt_text.startswith(question_text):
                raise direct_error
            prompt_indices = self._anchor_hidden_content_token_indices(
                text_inputs,
                prompt_text,
            )
            encoded = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping")
            if isinstance(offsets, torch.Tensor):
                offsets = offsets.detach().cpu().tolist()
            if not isinstance(offsets, (list, tuple)) or len(offsets) != len(prompt_indices):
                raise RuntimeError(
                    "attention-distribution gate could not align question offsets "
                    "with the prompt token span"
                ) from direct_error
            local_indices = [
                index
                for index, pair in enumerate(offsets)
                if isinstance(pair, (list, tuple))
                and len(pair) == 2
                and int(pair[1]) > 0
                and int(pair[0]) < len(question_text)
            ]
            if not local_indices:
                raise RuntimeError(
                    "attention-distribution gate found no prompt tokens overlapping the question"
                ) from direct_error
            return [prompt_indices[index] for index in local_indices]

    def _build_attention_distribution_features(
        self,
        *,
        q_entry: Dict[str, torch.Tensor],
        k_entry: Dict[str, torch.Tensor],
        query_indices: Sequence[int],
        frames: Sequence[Any],
        rotary_emb,
        layers,
        layer_idx: Optional[int] = None,
    ) -> Tuple[int, Dict[str, float]]:
        selected_layer = (
            self._task_gate_attention_layer(len(layers))
            if layer_idx is None
            else int(layer_idx)
        )
        q_raw = q_entry.get("q")
        k_raw = k_entry.get("k")
        q_positions = q_entry.get("positions")
        k_positions = k_entry.get("positions")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (q_raw, k_raw, q_positions, k_positions)
        ):
            raise RuntimeError(
                f"attention-distribution gate is missing Q/K/position capture at layer {selected_layer}"
            )
        attention_module = self._attention_module(layers[selected_layer])
        return selected_layer, self._attention_distribution_features_from_raw(
            q_raw=q_raw,
            k_raw=k_raw,
            q_positions=q_positions,
            k_positions=k_positions,
            query_indices=query_indices,
            unit_indices=[frame.token_indices for frame in frames],
            recent_unit_count=self._retrieval_recent_units(),
            rotary_emb=rotary_emb,
            attention_module=attention_module,
        )

    def _resolve_repo_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(os.getcwd(), path))

    def _load_task_gate_text_model(self) -> Optional[Dict[str, Any]]:
        if self._task_gate_mode() != "query_text_nb":
            return None
        path = str(self.config.get("task_gate_text_model_path", "") or "").strip()
        if not path:
            raise ValueError("task_gate_mode=query_text_nb requires task_gate_text_model_path.")
        model_path = self._resolve_repo_path(path)
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as exc:
            raise FileNotFoundError(f"Failed to read task gate text model: {model_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid task gate text model payload: {model_path}")
        if (
            payload.get("protocol") == "transductive_test_annotations"
            and not _as_bool(self.config, "allow_transductive_task_gate")
        ):
            raise ValueError(
                "Task gate was trained on OVO test annotations and is transductive. "
                "Use a train-split artifact for comparable results, or explicitly set "
                "allow_transductive_task_gate=true for a separately labelled upper-bound run."
            )
        weights = payload.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Task gate text model has no weights: {model_path}")
        payload["weights"] = {str(k): float(v) for k, v in weights.items()}
        payload["bias"] = float(payload.get("bias", 0.0))
        payload["threshold"] = float(payload.get("threshold", self.config.get("task_gate_text_threshold", 0.0)))
        payload["path"] = model_path
        return payload

    def _task_gate_features(self, prompt: str, max_ngram: int = 3) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", str(prompt).lower())
        features = set(tokens)
        if max_ngram >= 2:
            features.update(f"{tokens[i]}_{tokens[i + 1]}" for i in range(max(len(tokens) - 1, 0)))
        if max_ngram >= 3:
            features.update(
                f"{tokens[i]}_{tokens[i + 1]}_{tokens[i + 2]}"
                for i in range(max(len(tokens) - 2, 0))
            )
        return sorted(features)

    def _score_query_text_gate(self, prompt: str, source: str) -> Dict[str, Any]:
        model = self._task_gate_text_model
        if model is None:
            raise RuntimeError("query_text_nb gate model was not loaded.")
        max_ngram = int(model.get("max_ngram", 3) or 3)
        weights: Dict[str, float] = model["weights"]
        features = self._task_gate_features(prompt, max_ngram=max_ngram)
        matched = [(feature, weights[feature]) for feature in features if feature in weights]
        score = float(model["bias"] + sum(weight for _feature, weight in matched))
        threshold = float(self.config.get("task_gate_text_threshold", model.get("threshold", 0.0)))
        predicted = "backward" if score >= threshold else "realtime"
        return {
            "enabled": True,
            "mode": "query_text_nb",
            "source": source,
            "predicted_task_type": predicted,
            "selected_policy": "retrieval" if predicted == "backward" else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": score,
            "threshold": threshold,
            "matched_feature_count": len(matched),
            "top_matched_features": [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in sorted(matched, key=lambda item: abs(item[1]), reverse=True)[:12]
            ],
            "model_path": str(model.get("path", "")),
        }

    def _semantic_router_prompt(self, prompt: str) -> str:
        return (
            "You are routing an online video question to one of two memory policies.\n"
            "Choose BACKWARD if answering needs searching earlier video history, remembering prior events, "
            "or comparing against something that happened before the latest frames.\n"
            "Choose REALTIME if answering should rely on the latest/current visible frames near the question time.\n"
            "The question may contain multiple-choice answer options. Do not answer the question.\n"
            "Score only the route label for the question.\n\n"
            f"Question:\n{prompt}\n\nRoute label:"
        )

    def _choice_logits_router_prompt(self, prompt: str) -> str:
        return build_query_choice_router_prompt(
            prompt,
            normalize_query_choice_prompt_version(self.config),
        )

    def _choice_logits_token_id(self, label: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("query_choice_logits requires a tokenizer for A/B scoring.")
        retrieve_id, recent_id = resolve_query_choice_token_ids(tokenizer)
        if label == "A":
            return retrieve_id
        if label == "B":
            return recent_id
        raise ValueError(f"Unsupported query-choice label: {label!r}")

    def _score_query_choice_logits_gate(
        self,
        prompt: str,
        source: str,
        *,
        input_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        _resolved_text, resolved_input_source = resolve_task_gate_input(
            (
                self.config
                if input_source is None
                else {"task_gate_input_source": input_source}
            ),
            question_text=prompt,
            full_prompt=prompt,
        )
        router_prompt = self._choice_logits_router_prompt(prompt)
        inputs = self._build_text_inputs(router_prompt)
        tokenizer = self.processor.tokenizer
        retrieve_token_id, recent_token_id = resolve_query_choice_token_ids(tokenizer)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError("query_choice_logits expected rank-3 single-example model logits")

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            last_position = int(inputs["input_ids"].shape[1]) - 1
        else:
            valid_positions = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                raise RuntimeError("query_choice_logits received an empty router prompt")
            last_position = int(valid_positions[-1].item())

        next_token_logits = logits[0, last_position].float()
        retrieve_logit = float(next_token_logits[retrieve_token_id].item())
        recent_logit = float(next_token_logits[recent_token_id].item())
        choice_tokens = query_choice_token_strings(
            tokenizer,
            (retrieve_token_id, recent_token_id),
        )
        return build_query_choice_decision(
            retrieve_logit=retrieve_logit,
            recent_logit=recent_logit,
            retrieve_token_id=retrieve_token_id,
            recent_token_id=recent_token_id,
            choice_tokens=choice_tokens,
            router_prompt_version=normalize_query_choice_prompt_version(self.config),
            source=source,
            input_text=prompt,
            input_source=resolved_input_source,
            recent_policy=str(
                self.config.get("task_gate_realtime_policy", "recent_only")
            ),
            threshold=float(
                self.config.get("task_gate_query_choice_threshold", 0.0)
            ),
        )

    def _semantic_candidate_logprob(self, router_inputs: Dict[str, Any], candidate: str) -> Tuple[float, int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("query_semantic_zeroshot requires a tokenizer for candidate scoring.")

        input_ids = router_inputs["input_ids"][0]
        attention_mask = router_inputs.get("attention_mask")
        if attention_mask is not None:
            input_ids = input_ids[attention_mask[0].to(dtype=torch.bool)]

        candidate_ids = tokenizer.encode(str(candidate), add_special_tokens=False)
        if not candidate_ids:
            raise RuntimeError(f"Semantic gate candidate tokenized to an empty sequence: {candidate!r}")

        device = input_ids.device
        candidate_tensor = torch.tensor(candidate_ids, device=device, dtype=input_ids.dtype)
        full_ids = torch.cat([input_ids, candidate_tensor], dim=0).unsqueeze(0)
        full_attention = torch.ones_like(full_ids, device=device)
        prompt_len = int(input_ids.numel())
        target_len = int(candidate_tensor.numel())

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_attention,
                use_cache=False,
            )
        logits = outputs.logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target = candidate_tensor.to(dtype=torch.long).view(1, target_len, 1)
        token_log_probs = log_probs.gather(-1, target).squeeze(-1)
        return float(token_log_probs.sum().item() / max(target_len, 1)), target_len

    def _score_query_semantic_gate(self, prompt: str, source: str) -> Dict[str, Any]:
        router_prompt = self._semantic_router_prompt(prompt)
        inputs = self._build_text_inputs(router_prompt)
        label_scores = {
            "backward": self._semantic_candidate_logprob(inputs, "BACKWARD"),
            "realtime": self._semantic_candidate_logprob(inputs, "REALTIME"),
        }
        backward_logprob = float(label_scores["backward"][0])
        realtime_logprob = float(label_scores["realtime"][0])
        score = backward_logprob - realtime_logprob
        threshold = float(self.config.get("task_gate_text_threshold", 0.0))
        predicted = "backward" if score >= threshold else "realtime"
        max_logprob = max(backward_logprob, realtime_logprob)
        backward_weight = math.exp(backward_logprob - max_logprob)
        realtime_weight = math.exp(realtime_logprob - max_logprob)
        probability_denom = max(backward_weight + realtime_weight, 1e-12)
        return {
            "enabled": True,
            "mode": "query_semantic_zeroshot",
            "source": source,
            "predicted_task_type": predicted,
            "selected_policy": "retrieval" if predicted == "backward" else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": score,
            "threshold": threshold,
            "router_output": "label_completion_logprob",
            "label_logprobs": {
                "backward": backward_logprob,
                "realtime": realtime_logprob,
            },
            "label_token_counts": {
                label: int(values[1]) for label, values in label_scores.items()
            },
            "label_probabilities": {
                "backward": float(backward_weight / probability_denom),
                "realtime": float(realtime_weight / probability_denom),
            },
        }

    def _score_query_heuristic_gate(self, prompt: str, source: str) -> Dict[str, Any]:
        text = re.sub(r"\s+", " ", str(prompt).lower()).strip()
        if " options:" in text:
            question, options_text = text.split(" options:", 1)
        else:
            question, options_text = text, ""
        combined = f"{question} {options_text}"

        real_domain = re.compile(
            r"\b(monkey|giant|monster|gourd|bamboo|frog|shield|sword|bow|weapon|animal|bird|vehicle|"
            r"traffic|player|athlete|football|swimmer|surfboard|motorcycle|tire|bicycle|tool|wire|pipe|"
            r"basket|goal|bull|umpire|menu|screen|subtitle|sign|gpu|model|font|score|j1|j2|lane|pool|"
            r"brand|label|text|written|displayed|license|plate|time|date|number|jacket|shirt|package|"
            r"container|knife|drill|soil|seed|flower|pot|tree)\b"
        )
        house_domain = re.compile(
            r"\b(microwave|paper towel|plate|drawer|bowl|sink|book|bottle|tissue|stove|blender|bucket|"
            r"toolbox|cabinet|pillow|mirror|bedroom|bathroom|kitchen|stairs|staircase|couch|table|chair|"
            r"clock|teddy bear|rug|speaker|painting|piano|flowers|broom|hammer|phone|bag|towel|glasses|"
            r"dish washer|fridge|cupboard|counter|nightstand|door|wall|window|wreath|clothes)\b"
        )

        def _route(predicted: str, reason: str, score: float) -> Dict[str, Any]:
            return {
                "enabled": True,
                "mode": "query_heuristic_zeroshot",
                "source": source,
                "predicted_task_type": predicted,
                "selected_policy": "retrieval"
                if predicted == "backward"
                else str(self.config.get("task_gate_realtime_policy", "recent_only")),
                "score": float(score),
                "threshold": 0.0,
                "router_output": reason,
            }

        if "unable to answer" in options_text:
            return _route("backward", "hld_unable_option", 3.0)
        if re.search(r"\bwhat does the person do\b", question) and re.search(r"\b(before|after)\b", question):
            return _route("backward", "asi_person_before_after", 3.0)
        if re.search(r"\b(before|after)\b", question) and (
            "happened" in question or "which object" in question or "they" in question
        ):
            return _route("backward", "asi_temporal", 2.8)
        if re.search(r"\b(which object did the|what did the person|what did they|what happened before)\b", question):
            return _route("backward", "asi_event", 2.6)
        if re.search(r"\b(did|was|were|had)\s+(i|l)\b", question):
            return _route("backward", "first_person_past_aux", 2.0)
        if re.search(
            r"\b(i|l)\s+(picked|pick|put|placed|drop|dropped|pour|poured|pull|pulled|hold|held|talk|"
            r"interact|communicate|touch|move|moved|open|opened|close|closed|leave|left|take|took|"
            r"sharpen|warm|sit|see|saw)\b",
            question,
        ):
            return _route("backward", "first_person_event", 2.0)
        if "before i" in question or "before l" in question:
            return _route("backward", "before_i", 2.0)

        if re.search(r"\b(about to|going to|preparing to|planning to|getting ready|next)\b", question):
            return _route("realtime", "future_prediction", -3.0)
        if re.search(
            r"\b(text|word|written|displayed|label|name|number|score|time|date|license|plate|brand|"
            r"listed|read|say|subtitle|title|menu|font)\b",
            question,
        ):
            return _route("realtime", "ocr_text", -3.0)
        if re.search(
            r"\b(color|colour|pattern|patterns|texture|shape|material|wearing|expression|emotion|style|"
            r"hairstyle|made of|appearance|appear)\b",
            question,
        ) and not (house_domain.search(combined) and not real_domain.search(combined)):
            return _route("realtime", "attribute", -2.5)
        if re.search(
            r"\b(doing|action|being done|perform|interact|interacting|washing|washed|holding|held by|"
            r"used to|being used|manipulated|approaching|worked on|cut|chopped|gesture|technique|"
            r"flowing|evade|supporting)\b",
            question,
        ):
            return _route("realtime", "action_object", -2.5)
        if re.search(
            r"\b(relative|position|positions|direction|orientation|arrangement|arranged|relation|between|"
            r"closer|front|back|left|right|top|bottom|side|north|behind|above|along|facing|perspective|"
            r"where am i|where is the person|standing on|lying on)\b",
            question,
        ) and not (house_domain.search(combined) and not real_domain.search(combined)):
            return _route("realtime", "spatial", -2.0)
        if re.search(
            r"\b(what type of|what kind of|what animal|what object is being|what tool is being|what weapon|"
            r"what items are|what device|what part of|what can be seen|what seafood|which menu|which player|"
            r"which side|which location|which object is|which way|which of the players|which sheet|which spoon|"
            r"how is|how are|how would|describe the positions|can this)\b",
            question,
        ) and (real_domain.search(combined) or not house_domain.search(combined)):
            return _route("realtime", "expanded_realtime", -2.0)

        if re.search(r"\b(my|mine)\b", question):
            if real_domain.search(combined):
                return _route("realtime", "game_first_person", -1.5)
            return _route("backward", "first_person_possessive", 1.5)
        if re.search(r"\b(what is|what are|where is|where are|who is|how many)\b", question):
            if real_domain.search(combined) and not house_domain.search(combined):
                return _route("realtime", "present_domain_realtime", -1.5)
            if house_domain.search(combined) and not real_domain.search(combined):
                return _route("backward", "present_house_backward", 1.0)
            return _route("realtime", "present_question", -1.0)
        return _route("backward", "default_backward", 0.5)

    def _score_attention_skew_gate(self, selection: Dict[str, Any], source: str) -> Dict[str, Any]:
        scores = [float(score) for score, _frame in selection.get("short_scores", [])]
        threshold = float(self.config.get("task_gate_text_threshold", 0.0))
        if not scores:
            skew = 0.0
        else:
            mean = sum(scores) / len(scores)
            variance = sum((score - mean) ** 2 for score in scores) / max(len(scores), 1)
            skew = (max(scores) - mean) / math.sqrt(max(variance, 1e-12))
        predicted = "backward" if skew >= threshold else "realtime"
        return {
            "enabled": True,
            "mode": "attention_skew",
            "source": source,
            "predicted_task_type": predicted,
            "selected_policy": "retrieval" if predicted == "backward" else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": float(skew),
            "threshold": threshold,
            "candidate_count": len(scores),
        }

    def _anchor_kq_last_token_vector(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if states.dim() != 4 or int(states.shape[0]) != 1:
            raise RuntimeError(f"Unexpected q/k state shape for anchor KQ gate: {tuple(states.shape)}")
        seq_len = int(states.shape[2])
        if seq_len <= 0:
            raise RuntimeError("Anchor KQ gate received an empty q/k sequence.")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.numel() > 0:
            mask = attention_mask[0].to(device=states.device, dtype=torch.bool).view(-1)
            valid = torch.nonzero(mask, as_tuple=False).flatten()
            token_idx = int(valid[-1].item()) if int(valid.numel()) > 0 else seq_len - 1
        else:
            token_idx = seq_len - 1
        token_idx = max(0, min(token_idx, seq_len - 1))
        vector = states[0, :, token_idx, :].float().mean(dim=0)
        return torch.nn.functional.normalize(vector, dim=0).detach().cpu()

    def _anchor_hidden_content_token_indices(
        self,
        text_inputs: Dict[str, Any],
        raw_text: str,
    ) -> List[int]:
        input_ids = text_inputs.get("input_ids")
        tokenizer = getattr(self.processor, "tokenizer", None)
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
            raise RuntimeError("Anchor-Hidden gate requires rank-2 input_ids.")
        if tokenizer is None:
            raise RuntimeError("Anchor-Hidden gate requires a tokenizer.")
        templated_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        content_ids = [
            int(value)
            for value in tokenizer.encode(str(raw_text), add_special_tokens=False)
        ]
        if not content_ids:
            raise ValueError("Anchor-Hidden gate text tokenized to an empty sequence.")
        width = len(content_ids)
        starts = [
            start
            for start in range(len(templated_ids) - width + 1)
            if templated_ids[start : start + width] == content_ids
        ]
        if len(starts) != 1:
            raise RuntimeError(
                "Anchor-Hidden gate expected exactly one raw-text token span in "
                f"the chat template, found {len(starts)}."
            )
        return list(range(starts[0], starts[0] + width))

    def _anchor_hidden_content_mean_vector(
        self,
        hidden_states: torch.Tensor,
        token_indices: Sequence[int],
    ) -> torch.Tensor:
        if hidden_states.dim() != 3 or int(hidden_states.shape[0]) != 1:
            raise RuntimeError(
                "Unexpected hidden state shape for Anchor-Hidden gate: "
                f"{tuple(hidden_states.shape)}"
            )
        indices = [int(index) for index in token_indices]
        if not indices:
            raise RuntimeError("Anchor-Hidden gate received an empty content span.")
        if min(indices) < 0 or max(indices) >= int(hidden_states.shape[1]):
            raise RuntimeError(
                "Anchor-Hidden content span is outside the hidden-state sequence: "
                f"indices=[{min(indices)}, {max(indices)}], seq_len={int(hidden_states.shape[1])}."
            )
        index = torch.tensor(indices, device=hidden_states.device, dtype=torch.long)
        vector = hidden_states[0].index_select(0, index).float().mean(dim=0)
        return torch.nn.functional.normalize(vector, dim=0).detach().cpu()

    def _anchor_hidden_encode_text(self, text: str, probe_layer: int) -> torch.Tensor:
        probe_text = str(text).strip()
        if not probe_text:
            raise ValueError("Anchor-Hidden gate text must be non-empty.")
        text_inputs = self._build_text_inputs(probe_text)
        content_indices = self._anchor_hidden_content_token_indices(
            text_inputs,
            probe_text,
        )
        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(text_inputs, language)
        input_embeds = language_inputs["inputs_embeds"]
        layers, rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        probe_layer = int(probe_layer)
        if not 0 <= probe_layer < len(layers):
            raise ValueError(
                "Anchor-Hidden probe_layer must be in "
                f"[0, {len(layers) - 1}], got {probe_layer}"
            )
        positions = self._normalize_rope_positions(
            language_inputs.get("position_ids"),
            seq_len=int(input_embeds.shape[1]),
            device=input_embeds.device,
        )
        text_token_count = int(input_embeds.shape[1])
        with torch.no_grad():
            _hidden, raw_lower_kv = self._forward_lower_layers_raw(
                hidden_states=input_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=probe_layer + 1,
                past_raw_kv={},
                positions=positions,
                update_cache=True,
                capture_hidden_layers={probe_layer},
                # The standalone text probe has no video history. Treat its
                # complete causal prefix as the ReKV sink/local window.
                sink_len=text_token_count,
                local_window_tokens=text_token_count,
            )
        layer_cache = raw_lower_kv.get(probe_layer)
        hidden_states = layer_cache.get("hidden") if isinstance(layer_cache, dict) else None
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                f"Anchor-Hidden gate failed to capture hidden states at layer {probe_layer}."
            )
        return self._anchor_hidden_content_mean_vector(hidden_states, content_indices)

    def _anchor_hidden_layer(self, layer_count: int) -> int:
        configured = self.config.get("task_gate_anchor_hidden_layer")
        probe_layer = (
            _as_int(self.config, "prune_layer") - 1
            if configured is None or str(configured).strip() == ""
            else int(configured)
        )
        if not 0 <= probe_layer < int(layer_count):
            raise ValueError(
                "task_gate_anchor_hidden_layer must be in "
                f"[0, {int(layer_count) - 1}], got {probe_layer}"
            )
        return probe_layer

    def _ensure_anchor_hidden_vectors(self, probe_layer: int) -> Dict[str, torch.Tensor]:
        past_anchor = str(self.config.get("task_gate_past_anchor", "") or "").strip()
        nonpast_anchor = str(self.config.get("task_gate_nonpast_anchor", "") or "").strip()
        if not past_anchor or not nonpast_anchor:
            raise ValueError("task_gate_past_anchor and task_gate_nonpast_anchor must be non-empty.")
        signature = (int(probe_layer), past_anchor, nonpast_anchor)
        cached = getattr(self, "_task_gate_anchor_hidden", None)
        if cached is not None and getattr(self, "_task_gate_anchor_hidden_signature", None) == signature:
            return cached
        vectors = {
            "past": self._anchor_hidden_encode_text(past_anchor, probe_layer),
            "nonpast": self._anchor_hidden_encode_text(nonpast_anchor, probe_layer),
        }
        self._task_gate_anchor_hidden = vectors
        self._task_gate_anchor_hidden_signature = signature
        return vectors

    def _score_anchor_hidden_gate(
        self,
        prompt: str,
        source: str,
        *,
        input_source: str = "question_text",
    ) -> Dict[str, Any]:
        probe_text = str(prompt).strip()
        if not probe_text:
            raise ValueError("Anchor-Hidden gate requires a non-empty question prompt.")
        layers, _rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        probe_layer = self._anchor_hidden_layer(len(layers))
        anchors = self._ensure_anchor_hidden_vectors(probe_layer)
        question = self._anchor_hidden_encode_text(probe_text, probe_layer).float()
        past_cosine = float(torch.dot(question, anchors["past"].float()).item())
        nonpast_cosine = float(torch.dot(question, anchors["nonpast"].float()).item())
        threshold = float(self.config["task_gate_anchor_hidden_threshold"])
        return build_anchor_hidden_decision(
            past_cosine=past_cosine,
            nonpast_cosine=nonpast_cosine,
            threshold=threshold,
            probe_layer=probe_layer,
            source=source,
            input_text=prompt,
            input_source=input_source,
            recent_policy=str(
                self.config.get("task_gate_realtime_policy", "recent_only")
            ),
        )

    def _anchor_kq_encode_text(self, text: str, probe_layer: int) -> Dict[str, torch.Tensor]:
        probe_text = str(text).strip()
        if not probe_text:
            raise ValueError("Anchor KQ gate text must be non-empty.")
        text_inputs = self._build_text_inputs(probe_text)
        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(text_inputs, language)
        input_embeds = language_inputs["inputs_embeds"]
        layers, rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        probe_layer = int(probe_layer)
        if not 0 <= probe_layer < len(layers):
            raise ValueError(
                "Anchor-KQ probe_layer must be in "
                f"[0, {len(layers) - 1}], got {probe_layer}"
            )
        end_layer = probe_layer + 1
        positions = self._normalize_rope_positions(
            language_inputs.get("position_ids"),
            seq_len=int(input_embeds.shape[1]),
            device=input_embeds.device,
        )
        text_token_count = int(input_embeds.shape[1])
        with torch.no_grad():
            _hidden, raw_lower_kv = self._forward_lower_layers_raw(
                hidden_states=input_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=end_layer,
                past_raw_kv={},
                positions=positions,
                update_cache=True,
                capture_q_layers={probe_layer},
                # The standalone text probe has no video history. Treat its
                # complete causal prefix as the ReKV sink/local window so the
                # gate remains compatible with use_rekv_sink=true.
                sink_len=text_token_count,
                local_window_tokens=text_token_count,
            )
        layer_cache = raw_lower_kv.get(probe_layer)
        if not isinstance(layer_cache, dict):
            raise RuntimeError(f"Missing anchor KQ gate layer cache at layer {probe_layer}.")
        q_states = layer_cache.get("q")
        k_states = layer_cache.get("k")
        if not isinstance(q_states, torch.Tensor) or not isinstance(k_states, torch.Tensor):
            raise RuntimeError("Anchor KQ gate failed to capture q/k states.")
        attention_mask = text_inputs.get("attention_mask")
        return {
            "q": self._anchor_kq_last_token_vector(q_states, attention_mask),
            "k": self._anchor_kq_last_token_vector(k_states, attention_mask),
        }

    def _ensure_anchor_kq_vectors(self, probe_layer: int) -> Dict[str, torch.Tensor]:
        past_anchor = str(self.config.get("task_gate_past_anchor", "") or "").strip()
        nonpast_anchor = str(self.config.get("task_gate_nonpast_anchor", "") or "").strip()
        if not past_anchor or not nonpast_anchor:
            raise ValueError("task_gate_past_anchor and task_gate_nonpast_anchor must be non-empty.")
        signature = (int(probe_layer), past_anchor, nonpast_anchor)
        if self._task_gate_anchor_k is not None and self._task_gate_anchor_signature == signature:
            return self._task_gate_anchor_k
        past_vectors = self._anchor_kq_encode_text(past_anchor, probe_layer)
        nonpast_vectors = self._anchor_kq_encode_text(nonpast_anchor, probe_layer)
        self._task_gate_anchor_k = {
            "past": past_vectors["k"],
            "nonpast": nonpast_vectors["k"],
        }
        self._task_gate_anchor_signature = signature
        return self._task_gate_anchor_k

    def _anchor_kq_threshold(self) -> float:
        return float(self.config["task_gate_anchor_kq_threshold"])

    def _anchor_kq_layer(self, layer_count: int) -> int:
        configured = self.config.get("task_gate_anchor_kq_layer")
        probe_layer = (
            _as_int(self.config, "prune_layer") - 1
            if configured is None or str(configured).strip() == ""
            else int(configured)
        )
        if not 0 <= probe_layer < int(layer_count):
            raise ValueError(
                "task_gate_anchor_kq_layer must be in "
                f"[0, {int(layer_count) - 1}], got {probe_layer}"
            )
        return probe_layer

    def _score_anchor_kq_gate(self, prompt: str, source: str) -> Dict[str, Any]:
        probe_text = str(prompt).strip()
        if not probe_text:
            raise ValueError("Anchor KQ gate requires a non-empty question prompt.")
        layers, _rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        probe_layer = self._anchor_kq_layer(len(layers))
        anchors = self._ensure_anchor_kq_vectors(probe_layer)
        question_vectors = self._anchor_kq_encode_text(probe_text, probe_layer)
        question_q = torch.nn.functional.normalize(question_vectors["q"].float(), dim=0).detach().cpu()
        past_anchor_k = anchors["past"].float()
        nonpast_anchor_k = anchors.get("nonpast", anchors.get("current")).float()
        if int(past_anchor_k.numel()) != int(question_q.numel()) or int(nonpast_anchor_k.numel()) != int(question_q.numel()):
            raise RuntimeError(
                f"Anchor KQ gate vector mismatch: past_anchor_dim={int(past_anchor_k.numel())}, "
                f"nonpast_anchor_dim={int(nonpast_anchor_k.numel())}, question_dim={int(question_q.numel())}."
            )
        past_kq = float(torch.dot(past_anchor_k, question_q).item())
        nonpast_kq = float(torch.dot(nonpast_anchor_k, question_q).item())
        kq_diff = float(past_kq - nonpast_kq)
        threshold = self._anchor_kq_threshold()
        retrieval_enabled = bool(kq_diff >= threshold)
        predicted = "backward" if retrieval_enabled else "realtime"
        return {
            "enabled": True,
            "mode": "anchor_kq",
            "source": source,
            "predicted_task_type": predicted,
            "selected_policy": "retrieval"
            if retrieval_enabled
            else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": kq_diff,
            "threshold": threshold,
            "probe_layer": probe_layer,
            "kq_diff": kq_diff,
            "past_kq": past_kq,
            "nonpast_kq": nonpast_kq,
            # Deprecated output name retained for downstream readers.
            "current_kq": nonpast_kq,
            "retrieval_enabled": retrieval_enabled,
            "query_source": "question_text_only_last_token_head_mean",
            "pool": "last_token_head_mean",
            "rule": "enable_retrieval_if_past_kq_minus_nonpast_kq_gte_threshold",
        }

    def _prototype_probe_prompt(self, prompt: str, kind: str) -> str:
        if kind == "global":
            instruction = (
                "Routing probe: search broadly through the earlier video history. "
                "Prefer evidence that may appear before the latest frames."
            )
        elif kind == "local":
            instruction = (
                "Routing probe: focus on the latest current frames near the end of the observed video. "
                "Prefer immediate visual evidence."
            )
        else:
            raise ValueError(f"Unsupported probe prompt kind: {kind!r}")
        return f"{prompt}\n\n{instruction}"

    def _text_probe_query_vector(
        self,
        prompt: str,
        layers,
        rotary_emb,
        query_layer: int,
        past_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        position_start: int = 0,
        sink_len: int = 0,
        sink_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        local_window_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        text_inputs = self._build_text_inputs(prompt)
        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(text_inputs, language)
        input_embeds = language_inputs["inputs_embeds"]
        if past_raw_kv:
            positions = self._make_scalar_positions(int(position_start), int(input_embeds.shape[1]), input_embeds.device)
        else:
            positions = self._normalize_rope_positions(
                language_inputs.get("position_ids"),
                seq_len=int(input_embeds.shape[1]),
                device=input_embeds.device,
            )
        with torch.no_grad():
            _hidden, raw_cache = self._forward_lower_layers_raw(
                hidden_states=input_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=int(query_layer) + 1,
                past_raw_kv=past_raw_kv or {},
                positions=positions,
                update_cache=True,
                capture_q_layers={int(query_layer)},
                sink_len=int(sink_len),
                sink_raw_kv=sink_raw_kv,
                local_window_tokens=local_window_tokens,
            )
        layer_cache = raw_cache.get(int(query_layer))
        if not isinstance(layer_cache, dict) or not isinstance(layer_cache.get("q"), torch.Tensor):
            raise RuntimeError(f"Missing prototype probe Q capture at lower layer {query_layer}.")
        query = layer_cache["q"]
        key = layer_cache["k"]
        q_len = int(query.shape[2])
        keep = min(max(q_len, 1), 32)
        probe_positions = torch.arange(q_len - keep, q_len, device=query.device, dtype=torch.long)
        return self._normalized_query_vector(query, probe_positions, key_head_count=int(key.shape[1])).detach()

    def _task_gate_probe_vectors(
        self,
        prompt: str,
        layers,
        rotary_emb,
        query_layer: int,
        context_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        position_start: int = 0,
        context_sink_len: int = 0,
        context_sink_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        context_local_window_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if self._task_gate_mode() != "attention_probe":
            return {}
        if self._task_gate_probe_strategy() != "prototype_contrast":
            return {}
        return {
            "global": self._text_probe_query_vector(
                self._prototype_probe_prompt(prompt, "global"),
                layers=layers,
                rotary_emb=rotary_emb,
                query_layer=query_layer,
                past_raw_kv=context_raw_kv,
                position_start=position_start,
                sink_len=context_sink_len,
                sink_raw_kv=context_sink_raw_kv,
                local_window_tokens=context_local_window_tokens,
            ),
            "local": self._text_probe_query_vector(
                self._prototype_probe_prompt(prompt, "local"),
                layers=layers,
                rotary_emb=rotary_emb,
                query_layer=query_layer,
                past_raw_kv=context_raw_kv,
                position_start=position_start,
                sink_len=context_sink_len,
                sink_raw_kv=context_sink_raw_kv,
                local_window_tokens=context_local_window_tokens,
            ),
        }

    def _mean_topk(self, values: Sequence[float], k: int) -> float:
        if not values:
            return 0.0
        k = max(min(int(k), len(values)), 1)
        return float(sum(sorted((float(v) for v in values), reverse=True)[:k]) / k)

    def _probe_softmax_stats(
        self,
        entries: Sequence[Tuple[str, float, float]],
    ) -> Tuple[float, float, float, float, float]:
        if not entries:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        temperature = max(_as_float(self.config, "task_gate_probe_temperature"), 1e-6)
        max_score = max(float(score) for _kind, score, _pos in entries)
        weights = [math.exp((float(score) - max_score) / temperature) for _kind, score, _pos in entries]
        total = max(sum(weights), 1e-12)
        probs = [weight / total for weight in weights]
        recent_mass = sum(prob for prob, (kind, _score, _pos) in zip(probs, entries) if kind == "recent")
        nonrecent_mass = 1.0 - recent_mass
        center = sum(prob * float(pos) for prob, (_kind, _score, pos) in zip(probs, entries))
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        entropy_norm = entropy / math.log(max(len(entries), 2))
        topk = max(min(_as_int(self.config, "retrieval_topk_units"), len(entries)), 1)
        top_entries = sorted(zip(entries, probs), key=lambda item: item[0][1], reverse=True)[:topk]
        topk_nonrecent_fraction = sum(1 for (kind, _score, _pos), _prob in top_entries if kind != "recent") / float(topk)
        return float(recent_mass), float(nonrecent_mass), float(center), float(entropy_norm), float(topk_nonrecent_fraction)

    def _probe_topk_temporal_stats(
        self,
        entries: Sequence[Tuple[str, float, float]],
    ) -> Dict[str, float]:
        if not entries:
            return {
                "top1_recency": 1.0,
                "top1_is_recent": 0.0,
                "top1_is_search": 0.0,
                "top1_is_cluster": 0.0,
                "topk_center_recency": 1.0,
                "topk_early_fraction": 0.0,
                "topk_mid_fraction": 0.0,
                "topk_late_fraction": 0.0,
                "topk_recent_fraction": 0.0,
                "topk_search_fraction": 0.0,
                "topk_cluster_fraction": 0.0,
            }
        topk = max(min(_as_int(self.config, "retrieval_topk_units"), len(entries)), 1)
        top_entries = sorted(entries, key=lambda item: item[1], reverse=True)[:topk]
        top1_kind, _top1_score, top1_pos = top_entries[0]
        top_positions = [float(pos) for _kind, _score, pos in top_entries]
        return {
            "top1_recency": float(top1_pos),
            "top1_is_recent": 1.0 if top1_kind == "recent" else 0.0,
            "top1_is_search": 1.0 if top1_kind == "search" else 0.0,
            "top1_is_cluster": 1.0 if top1_kind == "cluster" else 0.0,
            "topk_center_recency": float(sum(top_positions) / len(top_positions)),
            "topk_early_fraction": sum(1 for pos in top_positions if pos <= 0.70) / float(topk),
            "topk_mid_fraction": sum(1 for pos in top_positions if 0.70 < pos < 0.90) / float(topk),
            "topk_late_fraction": sum(1 for pos in top_positions if pos >= 0.90) / float(topk),
            "topk_recent_fraction": sum(1 for kind, _score, _pos in top_entries if kind == "recent") / float(topk),
            "topk_search_fraction": sum(1 for kind, _score, _pos in top_entries if kind == "search") / float(topk),
            "topk_cluster_fraction": sum(1 for kind, _score, _pos in top_entries if kind == "cluster") / float(topk),
        }

    def _probe_score_entries(
        self,
        selection: Dict[str, Any],
        query_vec: torch.Tensor,
    ) -> List[Tuple[str, float, float]]:
        recent_frames = list(selection.get("recent", []))
        search_frames = list(selection.get("search_candidates", []))
        clusters = list(selection.get("all_clusters", []))
        max_frame_id = 0
        frame_like = list(recent_frames) + list(search_frames)
        if frame_like:
            max_frame_id = max(max_frame_id, max(int(frame.frame_id) for frame in frame_like))
        if clusters:
            max_frame_id = max(max_frame_id, max(int(cluster.end_frame) for cluster in clusters))
        denom = max(float(max_frame_id), 1.0)

        def _frame_score(frame: _FrameKVState) -> float:
            return float(torch.dot(frame.key_vec.float(), query_vec.float()).item())

        def _cluster_score(cluster: _LongKVCluster) -> float:
            return float(torch.dot(cluster.key_vec.float(), query_vec.float()).item())

        entries: List[Tuple[str, float, float]] = []
        for frame in recent_frames:
            entries.append(("recent", _frame_score(frame), float(frame.frame_id) / denom))
        for frame in search_frames:
            entries.append(("search", _frame_score(frame), float(frame.frame_id) / denom))
        for cluster in clusters:
            midpoint = (float(cluster.start_frame) + float(cluster.end_frame)) / 2.0
            entries.append(("cluster", _cluster_score(cluster), midpoint / denom))
        return entries

    def _score_vector_from_entries(self, entries: Sequence[Tuple[str, float, float]]) -> List[float]:
        return [float(score) for _kind, score, _pos in entries]

    def _centered_cosine(self, left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        left_mean = sum(float(value) for value in left) / len(left)
        right_mean = sum(float(value) for value in right) / len(right)
        left_centered = [float(value) - left_mean for value in left]
        right_centered = [float(value) - right_mean for value in right]
        numerator = sum(a * b for a, b in zip(left_centered, right_centered))
        left_norm = math.sqrt(sum(a * a for a in left_centered))
        right_norm = math.sqrt(sum(b * b for b in right_centered))
        denom = max(left_norm * right_norm, 1e-12)
        return float(numerator / denom)

    def _softmax_probs(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        temperature = max(_as_float(self.config, "task_gate_probe_temperature"), 1e-6)
        max_score = max(float(score) for score in scores)
        weights = [math.exp((float(score) - max_score) / temperature) for score in scores]
        total = max(sum(weights), 1e-12)
        return [float(weight / total) for weight in weights]

    def _js_similarity(self, left_scores: Sequence[float], right_scores: Sequence[float]) -> float:
        if len(left_scores) != len(right_scores) or not left_scores:
            return 0.0
        left = self._softmax_probs(left_scores)
        right = self._softmax_probs(right_scores)
        mid = [(a + b) / 2.0 for a, b in zip(left, right)]

        def _kl(p: Sequence[float], q: Sequence[float]) -> float:
            return sum(float(a) * math.log(max(float(a), 1e-12) / max(float(b), 1e-12)) for a, b in zip(p, q))

        # Higher is better: 1.0 means identical distributions, lower values are less similar.
        js = 0.5 * _kl(left, mid) + 0.5 * _kl(right, mid)
        return float(1.0 - js)

    def _prototype_contrast_features(self, selection: Dict[str, Any]) -> Dict[str, float]:
        query_vec = selection.get("query_vec")
        probe_vecs = selection.get("probe_query_vecs") or {}
        if not isinstance(query_vec, torch.Tensor):
            return {}
        global_vec = probe_vecs.get("global") if isinstance(probe_vecs, dict) else None
        local_vec = probe_vecs.get("local") if isinstance(probe_vecs, dict) else None
        if not isinstance(global_vec, torch.Tensor) or not isinstance(local_vec, torch.Tensor):
            return {}
        actual_scores = self._score_vector_from_entries(self._probe_score_entries(selection, query_vec))
        global_scores = self._score_vector_from_entries(self._probe_score_entries(selection, global_vec))
        local_scores = self._score_vector_from_entries(self._probe_score_entries(selection, local_vec))
        corr_global = self._centered_cosine(actual_scores, global_scores)
        corr_local = self._centered_cosine(actual_scores, local_scores)
        js_global = self._js_similarity(actual_scores, global_scores)
        js_local = self._js_similarity(actual_scores, local_scores)
        return {
            "prototype_corr_global": float(corr_global),
            "prototype_corr_local": float(corr_local),
            "prototype_corr_margin": float(corr_global - corr_local),
            "prototype_js_global": float(js_global),
            "prototype_js_local": float(js_local),
            "prototype_js_margin": float(js_global - js_local),
        }

    def _attention_probe_features(self, selection: Dict[str, Any]) -> Dict[str, float]:
        query_vec = selection.get("query_vec")
        if not isinstance(query_vec, torch.Tensor):
            return {}

        recent_frames = list(selection.get("recent", []))
        search_frames = list(selection.get("search_candidates", []))
        clusters = list(selection.get("all_clusters", []))
        entries = self._probe_score_entries(selection, query_vec)
        recent_scores = [score for kind, score, _pos in entries if kind == "recent"]
        search_scores = [score for kind, score, _pos in entries if kind == "search"]
        cluster_scores = [score for kind, score, _pos in entries if kind == "cluster"]
        nonrecent_scores = search_scores + cluster_scores

        recent_best = max(recent_scores) if recent_scores else 0.0
        search_best = max(search_scores) if search_scores else 0.0
        cluster_best = max(cluster_scores) if cluster_scores else 0.0
        nonrecent_best = max(nonrecent_scores) if nonrecent_scores else 0.0
        recent_mean_topk = self._mean_topk(recent_scores, self._retrieval_recent_units())
        nonrecent_mean_topk = self._mean_topk(nonrecent_scores, self._retrieval_topk_units())
        all_scores = recent_scores + nonrecent_scores
        if all_scores:
            mean = sum(all_scores) / len(all_scores)
            variance = sum((score - mean) ** 2 for score in all_scores) / max(len(all_scores), 1)
            score_skew = (max(all_scores) - mean) / math.sqrt(max(variance, 1e-12))
        else:
            score_skew = 0.0
        recent_mass, nonrecent_mass, center, entropy_norm, topk_nonrecent_fraction = self._probe_softmax_stats(entries)
        topk_temporal = self._probe_topk_temporal_stats(entries)
        selected_retrieved = list(selection.get("retrieved", []))
        retrieval_selected_frac = len(selected_retrieved) / float(max(len(search_frames), 1))

        features = {
            "recent_best": float(recent_best),
            "search_best": float(search_best),
            "cluster_best": float(cluster_best),
            "nonrecent_best": float(nonrecent_best),
            "recent_mean_topk": float(recent_mean_topk),
            "nonrecent_mean_topk": float(nonrecent_mean_topk),
            "margin_best": float(nonrecent_best - recent_best),
            "margin_topk": float(nonrecent_mean_topk - recent_mean_topk),
            "recent_mass": float(recent_mass),
            "nonrecent_mass": float(nonrecent_mass),
            "center_of_mass_recency": float(center),
            "entropy_norm": float(entropy_norm),
            "topk_nonrecent_fraction": float(topk_nonrecent_fraction),
            "score_skew": float(score_skew),
            "retrieval_selected_frac": float(retrieval_selected_frac),
            "recent_count": float(len(recent_frames)),
            "search_count": float(len(search_frames)),
            "cluster_count": float(len(clusters)),
        }
        features.update(topk_temporal)
        features.update(self._prototype_contrast_features(selection))
        return features

    def _score_attention_probe_gate(self, selection: Dict[str, Any], source: str) -> Dict[str, Any]:
        features = self._attention_probe_features(selection)
        strategy = self._task_gate_probe_strategy()
        if strategy == "prototype_contrast":
            score = (
                1.00 * float(features.get("prototype_corr_margin", 0.0))
                + 0.35 * float(features.get("prototype_js_margin", 0.0))
            )
        else:
            score = (
                0.55 * float(features.get("topk_early_fraction", 0.0))
                + 0.35 * float(features.get("topk_cluster_fraction", 0.0))
                + 0.25 * float(0.78 - features.get("topk_center_recency", 0.78))
                + 0.15 * float(1.0 - features.get("top1_recency", 1.0))
                + 0.08 * float(features.get("top1_is_cluster", 0.0))
                - 0.55 * float(features.get("topk_late_fraction", 0.0))
                - 0.25 * float(features.get("topk_recent_fraction", 0.0))
                - 0.08 * float(features.get("score_skew", 0.0) - 2.0)
            )
        threshold = float(self.config.get("task_gate_probe_threshold", 0.0))
        predicted = "backward" if score >= threshold else "realtime"
        return {
            "enabled": True,
            "mode": "attention_probe",
            "source": source,
            "predicted_task_type": predicted,
            "selected_policy": "retrieval" if predicted == "backward" else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": float(score),
            "threshold": float(threshold),
            "scoring": f"training_free_{strategy}",
            "features": {name: float(value) for name, value in sorted(features.items())},
        }

    def _score_attention_distribution_gate(
        self,
        selection: Dict[str, Any],
        prompt: str,
        source: str,
    ) -> Dict[str, Any]:
        features = selection.get("attention_distribution_features")
        if not isinstance(features, dict):
            raise RuntimeError(
                "attention_distribution gate is missing matching-head attention features"
            )
        question_text = str(
            getattr(self.owner, "_task_gate_query_text", "") or ""
        ).strip()
        semantic = self._score_anchor_hidden_gate(
            question_text or prompt,
            source=source,
        )
        semantic_score = float(semantic["retrieval_score"])
        classifier = self.config.get("task_gate_attention_classifier") or {}
        classifier_metadata = None
        if classifier:
            raw_values = {
                "semantic_score": semantic_score,
                # A high deficit means the shallow pass allocated less
                # history attention than a cardinality-matched uniform prior.
                "history_attention_deficit": -float(features["history_enrichment"]),
                # Low dispersion makes the deficit consistent rather than an
                # outlier from one head or one question token.
                "head_dispersion_deficit": -float(features["head_enrichment_std"]),
                "query_dispersion_deficit": -float(features["query_enrichment_std"]),
            }
            specifications = classifier.get("features")
            if not isinstance(specifications, list) or not specifications:
                raise ValueError(
                    "task_gate_attention_classifier.features must be a non-empty list"
                )
            score = float(classifier.get("bias", 0.0))
            contributions = {}
            calibration = {}
            for specification in specifications:
                if not isinstance(specification, dict):
                    raise ValueError(
                        "task_gate_attention_classifier feature entries must be objects"
                    )
                name = str(specification.get("name", ""))
                if name not in raw_values:
                    raise ValueError(
                        f"unsupported attention classifier feature: {name!r}"
                    )
                mean = float(specification["mean"])
                scale = float(specification["scale"])
                weight = float(specification["weight"])
                if scale <= 0:
                    raise ValueError(
                        f"attention classifier scale for {name} must be positive"
                    )
                standardized = (raw_values[name] - mean) / scale
                contribution = weight * standardized
                score += contribution
                contributions[name] = float(contribution)
                calibration[name] = {
                    "mean": mean,
                    "scale": scale,
                    "weight": weight,
                }
            attention_score = float(
                sum(
                    contribution
                    for name, contribution in contributions.items()
                    if name != "semantic_score"
                )
            )
            attention_weight = None
            classifier_metadata = {
                "kind": str(classifier.get("kind", "standardized_linear")),
                "bias": float(classifier.get("bias", 0.0)),
                "raw_values": raw_values,
                "contributions": contributions,
                "calibration": calibration,
            }
        else:
            attention_score = float(features.get("history_enrichment", 0.0))
            attention_weight = float(self.config["task_gate_attention_weight"])
            score = semantic_score + attention_weight * attention_score
        threshold = float(self.config["task_gate_attention_threshold"])
        retrieval_enabled = bool(score >= threshold)
        return {
            "enabled": True,
            "mode": "attention_distribution",
            "source": source,
            "predicted_task_type": "backward" if retrieval_enabled else "realtime",
            "selected_policy": "retrieval"
            if retrieval_enabled
            else str(self.config.get("task_gate_realtime_policy", "recent_only")),
            "score": float(score),
            "retrieval_score": float(score),
            "threshold": float(threshold),
            "retrieval_enabled": retrieval_enabled,
            "probe_layer": int(selection["attention_distribution_layer"]),
            "semantic_probe_layer": int(semantic["probe_layer"]),
            "semantic_score": semantic_score,
            "attention_score": attention_score,
            "attention_weight": attention_weight,
            "classifier": classifier_metadata,
            "features": {
                name: float(value) for name, value in sorted(features.items())
            },
            "observation_features": {
                str(layer): {
                    name: float(value)
                    for name, value in sorted(layer_features.items())
                }
                for layer, layer_features in sorted(
                    (selection.get("attention_distribution_observation_features") or {}).items()
                )
            },
            "query_source": "question_text_content_q_to_video_k",
            "representation": "matching_head_scaled_qk_visual_attention",
            "rule": (
                "enable_retrieval_if_standardized_semantic_plus_attention_deficit_gte_threshold"
                if classifier_metadata is not None
                else "enable_retrieval_if_semantic_prior_plus_attention_enrichment_gte_threshold"
            ),
        }

    def _score_history_layer_decay_gate(
        self,
        selection: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        threshold = float(
            self.config.get("task_gate_history_decay_threshold", 0.0)
        )
        observations = selection.get("attention_distribution_observation_features")
        if not isinstance(observations, dict):
            raise RuntimeError(
                "history_layer_decay gate is missing layerwise attention observations"
            )
        available = {
            int(layer): float(features["history_attention_mass"])
            for layer, features in observations.items()
            if isinstance(features, dict)
            and float(features.get("attention_available", 0.0)) > 0.0
        }
        if len(available) < 2:
            decision = {
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

    def _set_gate_decision(self, decision: Dict[str, Any]) -> None:
        self.last_gate_decision = decision
        self.owner.last_gate_decision = decision

    @staticmethod
    def _attach_recent_retrieval_unit_scores(
        decision: Dict[str, Any],
        selection: Dict[str, Any],
    ) -> None:
        observation = selection.get("recent_retrieval_unit_scores")
        if isinstance(observation, dict):
            decision["recent_retrieval_unit_scores"] = observation

    def _retrieval_result_diagnostics(
        self,
        selection: Dict[str, Any],
    ) -> Dict[str, Any]:
        stats = dict(selection.get("token_selection_stats") or {})
        strategy = str(
            stats.get(
                "retrieval_score_strategy",
                self.config.get("retrieval_score_strategy", "shallow_unit_cosine"),
            )
        )
        vote_scores = {
            int(item["frame_id"]): item
            for item in stats.get("retrieval_vote_unit_scores", [])
            if isinstance(item, dict) and "frame_id" in item
        }
        shallow_scores = {
            int(frame.frame_id): float(score)
            for score, frame in selection.get("short_scores", [])
        }
        selected_units = []
        for rank, frame in enumerate(selection.get("retrieved_seed", []), start=1):
            frame_id = int(frame.frame_id)
            vote = vote_scores.get(frame_id, {})
            selected_units.append(
                {
                    "rank": rank,
                    "frame_id": frame_id,
                    "sample_index": int(frame.sample_index),
                    "timestamp": float(frame.timestamp),
                    "source_sample_indices": [
                        int(index) for index in (frame.source_sample_indices or [])
                    ],
                    "source_timestamps": [
                        float(timestamp) for timestamp in (frame.source_timestamps or [])
                    ],
                    "score": shallow_scores.get(frame_id),
                    "votes": vote.get("votes"),
                    "attention_tiebreak": vote.get("attention_tiebreak"),
                }
            )
        diagnostics = {
            "schema_version": 1,
            "strategy": strategy,
            "selected_units": selected_units,
            "recent_unit_ids": [
                int(frame.frame_id) for frame in selection.get("recent", [])
            ],
        }
        if strategy == "shallow_layer_token_vote":
            diagnostics.update(
                {
                    "layer_indices": stats.get("retrieval_vote_layer_indices", []),
                    "topk_tokens_per_layer": stats.get(
                        "retrieval_vote_topk_tokens_per_layer"
                    ),
                    "query_token_mode": stats.get(
                        "retrieval_vote_query_token_mode", "all_mean"
                    ),
                    "query_token_count": stats.get("retrieval_vote_query_token_count"),
                    "diversity_mode": stats.get("retrieval_vote_diversity_mode", "off"),
                    "diversity_pool_size": stats.get(
                        "retrieval_vote_diversity_pool_size"
                    ),
                }
            )
        return diagnostics

    def _apply_task_gate(
        self,
        selection: Dict[str, Any],
        prompt: str,
        source: str,
    ) -> Dict[str, Any]:
        mode = self._task_gate_mode()
        if mode == "off":
            decision = {
                "enabled": False,
                "mode": "off",
                "source": source,
                "predicted_task_type": "unknown",
                "selected_policy": "retrieval",
                "retrieval_diagnostics": self._retrieval_result_diagnostics(selection),
            }
            self._attach_recent_retrieval_unit_scores(decision, selection)
            self._set_gate_decision(decision)
            return selection

        if mode == "recent_context_sufficiency":
            decision = dict(self.last_gate_decision or {})
            if decision.get("mode") != mode:
                raise RuntimeError(
                    "recent-context sufficiency decision must run before retrieval"
                )
            if decision.get("selected_policy") != "retrieval":
                raise RuntimeError(
                    "recent-only sufficiency decisions must answer before retrieval"
                )
            decision["retrieval_diagnostics"] = self._retrieval_result_diagnostics(
                selection
            )
            gated = dict(selection)
            gated["token_selection_stats"] = dict(
                gated.get("token_selection_stats") or {}
            )
            gated["token_selection_stats"].update(
                {
                    "task_gate_mode": mode,
                    "task_gate_predicted_task_type": decision.get(
                        "predicted_task_type"
                    ),
                    "task_gate_selected_policy": "retrieval",
                    "task_gate_score": decision.get("score"),
                    "task_gate_threshold": decision.get("threshold"),
                }
            )
            self._set_gate_decision(decision)
            return gated

        if "options:" not in str(prompt).lower() and mode == "query_text_nb":
            decision = {
                "enabled": True,
                "mode": mode,
                "source": source,
                "predicted_task_type": "unknown",
                "selected_policy": "retrieval",
                "reason": "prompt_has_no_options",
            }
            self._attach_recent_retrieval_unit_scores(decision, selection)
            self._set_gate_decision(decision)
            return selection

        if mode == "decision_replay":
            decision = self._score_task_gate_replay(source=source)
        elif mode == "latest_unit_score":
            decision = self._score_latest_unit_gate(selection, source=source)
        elif mode in {"anchor_hidden", "anchor_kq"}:
            question_text = str(
                getattr(self.owner, "_task_gate_query_text", "") or ""
            ).strip()
            gate_text, input_source = resolve_task_gate_input(
                self.config,
                question_text=question_text,
                full_prompt=prompt,
            )
            if mode == "anchor_hidden":
                decision = self._score_anchor_hidden_gate(
                    gate_text,
                    source=source,
                    input_source=input_source,
                )
            else:
                decision = self._score_anchor_kq_gate(
                    gate_text,
                    source=source,
                )
                decision["query_source"] = query_source_label(
                    input_source,
                    "last_token_head_mean",
                )
                decision.update(task_gate_input_metadata(gate_text, input_source))
        elif mode == "query_text_nb":
            decision = self._score_query_text_gate(prompt, source=source)
        elif mode == "query_choice_logits":
            question_text = str(
                getattr(self.owner, "_task_gate_query_text", "") or ""
            ).strip()
            gate_text, input_source = resolve_task_gate_input(
                self.config,
                question_text=question_text,
                full_prompt=prompt,
            )
            decision = self._score_query_choice_logits_gate(
                gate_text,
                source=source,
                input_source=input_source,
            )
        elif mode == "query_semantic_zeroshot":
            decision = self._score_query_semantic_gate(prompt, source=source)
        elif mode == "query_heuristic_zeroshot":
            decision = self._score_query_heuristic_gate(prompt, source=source)
        elif mode == "attention_distribution":
            decision = self._score_attention_distribution_gate(
                selection,
                prompt=prompt,
                source=source,
            )
        elif mode == "history_layer_decay":
            decision = self._score_history_layer_decay_gate(selection, source=source)
        elif mode == "attention_probe":
            decision = self._score_attention_probe_gate(selection, source=source)
        else:
            decision = self._score_attention_skew_gate(selection, source=source)

        gated = selection
        if decision.get("selected_policy") == "recent_only":
            gated = dict(selection)
            recent_frames = list(selection.get("recent", []))
            realtime_recent_units = self.config.get("task_gate_realtime_recent_units")
            if realtime_recent_units is not None:
                realtime_recent_units = int(realtime_recent_units)
                if realtime_recent_units < 0:
                    raise ValueError("task_gate_realtime_recent_units must be non-negative")
                recent_frames = (
                    recent_frames[-realtime_recent_units:]
                    if realtime_recent_units > 0
                    else []
                )
            gated["short"] = recent_frames
            gated["recent"] = recent_frames
            gated["clusters"] = []
            gated["retrieved"] = []
            gated["retrieved_seed"] = []
            gated["token_selection_stats"] = dict(
                gated.get("token_selection_stats") or {}
            )
            gated["token_selection_stats"].update(
                {
                    "task_gate_mode": decision.get("mode"),
                    "task_gate_predicted_task_type": decision.get("predicted_task_type"),
                    "task_gate_selected_policy": decision.get("selected_policy"),
                    "task_gate_score": decision.get("score"),
                    "task_gate_threshold": decision.get("threshold"),
                    "task_gate_realtime_recent_units": realtime_recent_units,
                }
            )
        else:
            gated = (
                self._rerank_siglip_evidence(selection, prompt=prompt)
                if self._evidence_retrieval_backend() == "siglip"
                else dict(selection)
            )
            gated["token_selection_stats"] = dict(
                gated.get("token_selection_stats") or {}
            )
            if isinstance(gated.get("evidence_retrieval"), dict):
                decision["evidence_retrieval"] = dict(gated["evidence_retrieval"])
                gated["token_selection_stats"].update(
                    {
                        "retrieval_selection_granularity": "unit",
                        "evidence_retrieval_backend": "siglip",
                        "evidence_retrieval_query_source": gated["evidence_retrieval"]["query_source"],
                        "evidence_retrieval_model_path": gated["evidence_retrieval"]["model_path"],
                        "evidence_retrieval_candidate_count": gated["evidence_retrieval"]["candidate_count"],
                        "evidence_retrieval_seed_units": gated["evidence_retrieval"]["retrieved_seed_unit_ids"],
                    }
                )
            gated["token_selection_stats"].update(
                {
                    "task_gate_mode": decision.get("mode"),
                    "task_gate_predicted_task_type": decision.get("predicted_task_type"),
                    "task_gate_selected_policy": decision.get("selected_policy"),
                    "task_gate_score": decision.get("score"),
                    "task_gate_threshold": decision.get("threshold"),
                }
            )
        decision["retrieval_selection"] = {
            "expansion_strategy": _retrieval_expansion_strategy(self.config),
            "seed_unit_ids": [
                int(getattr(frame, "frame_id", frame))
                for frame in gated.get("retrieved_seed", [])
            ],
            "reference_expanded_unit_ids": [
                int(frame_id)
                for frame_id in gated.get(
                    "retrieval_reference_expanded_unit_ids", []
                )
            ],
            "retrieved_unit_ids": [
                int(getattr(frame, "frame_id", frame))
                for frame in gated.get("retrieved", [])
            ],
            "recent_unit_ids": [
                int(getattr(frame, "frame_id", frame))
                for frame in gated.get("recent", [])
            ],
            "selected_short_unit_ids": [
                int(getattr(frame, "frame_id", frame))
                for frame in gated.get("short", [])
            ],
        }
        self._attach_recent_retrieval_unit_scores(decision, selection)
        self._set_gate_decision(decision)
        if _as_bool(self.config, "debug"):
            print(f"[{MODEL_NAME}] task_gate_decision={json.dumps(decision, ensure_ascii=False)}", flush=True)
        return gated
