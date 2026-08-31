"""Task gates for the LLaVA-OneVision runtime."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch

from src.shallowstream.common import expand_temporal_neighbors
from src.shallowstream.evidence_retrieval import (
    evidence_retrieval_backend,
    get_siglip_evidence_retriever,
    resolve_evidence_query,
)
from src.shallowstream.task_gate import (
    TASK_GATE_INPUT_SOURCES,
    build_anchor_hidden_decision,
    build_query_choice_decision,
    build_query_choice_router_prompt,
    normalize_query_choice_prompt_version,
    query_choice_token_strings,
    resolve_query_choice_token_ids,
    resolve_task_gate_input,
)


class OneVisionTaskGateMixin:
    def _evidence_retrieval_backend(self) -> str:
        return evidence_retrieval_backend(self.config)

    def _select_siglip_evidence_frames(
        self,
        *,
        full_prompt: str,
        question_text: str,
    ) -> tuple[List[int], Dict[str, Any]]:
        frame_spans = self.state.get("frame_spans")
        evidence_images = self.state.get("frame_evidence_images")
        if not isinstance(frame_spans, list) or not isinstance(evidence_images, list):
            raise RuntimeError("OneVision SigLIP evidence images are not initialized")
        total_frames = len(frame_spans)
        if len(evidence_images) != total_frames:
            raise RuntimeError(
                "OneVision SigLIP evidence image/frame count mismatch: "
                f"{len(evidence_images)} != {total_frames}"
            )
        query, query_source = resolve_evidence_query(
            self.config,
            full_prompt=full_prompt,
            question_text=question_text,
        )
        candidate_start = self._short_window_start(total_frames)
        candidate_idx = list(range(candidate_start, total_frames))
        recent_n = min(
            max(0, int(self.config.get("retrieval_recent_frames", 0))),
            len(candidate_idx),
        )
        recent_idx = candidate_idx[-recent_n:] if recent_n > 0 else []
        important_candidates = (
            candidate_idx[:-recent_n] if recent_n > 0 else candidate_idx
        )
        result = get_siglip_evidence_retriever(self).score(
            query,
            [evidence_images[index] for index in important_candidates],
        )
        if len(result.scores) != len(important_candidates):
            raise RuntimeError("OneVision SigLIP score count does not match candidate frames")
        scored = sorted(
            zip(result.scores, important_candidates),
            key=lambda item: (-float(item[0]), int(item[1])),
        )
        topk = min(
            max(0, int(self.config.get("retrieval_topk_frames", 0))),
            len(scored),
        )
        seed_idx = [int(index) for _score, index in scored[:topk]]
        important_idx = expand_temporal_neighbors(
            seed_idx,
            candidate_idx,
            previous=max(0, int(self.config.get("retrieval_expand_prev_frames", 0))),
            following=max(0, int(self.config.get("retrieval_expand_next_frames", 0))),
            previous_stride=max(1, int(self.config.get("retrieval_expand_prev_stride", 1))),
            following_stride=max(1, int(self.config.get("retrieval_expand_next_stride", 1))),
        )
        keep_idx = sorted(set(recent_idx).union(important_idx))
        score_by_idx = {int(index): float(score) for score, index in scored}
        source_ids = self.state.get("frame_source_ids")
        metadata = dict(result.metadata)
        metadata.update(
            {
                "query_source": query_source,
                "candidate_frame_ids": [int(index) for index in important_candidates],
                "candidate_scores": [
                    float(score_by_idx[index]) for index in important_candidates
                ],
                "retrieved_seed_frame_ids": seed_idx,
                "retrieved_expanded_frame_ids": [int(index) for index in important_idx],
                "recent_frame_ids": [int(index) for index in recent_idx],
            }
        )
        self.state["last_selection"] = {
            "keep_idx": [int(index) for index in keep_idx],
            "recent_idx": [int(index) for index in recent_idx],
            "important_seed_idx": seed_idx,
            "important_expanded_idx": [int(index) for index in important_idx],
            "long_cluster_indices": [],
            "long_cluster_scores": [],
            "selected_scores": [
                float(score_by_idx[index])
                for index in keep_idx
                if index in score_by_idx
            ],
            "selected_source_ts_s": [
                float(source_ids[index])
                for index in keep_idx
                if isinstance(source_ids, list) and index < len(source_ids)
            ],
            "score_order": "highest",
            "evidence_retrieval": metadata,
        }
        return keep_idx, metadata

    def _task_gate_mode(self) -> str:
        mode = str(self.config.get("task_gate_mode", "off") or "off").strip().lower()
        supported = {
            "off",
            "anchor_hidden",
            "history_layer_decay",
            "latest_unit_score",
            "query_choice_logits",
        }
        if mode not in supported:
            raise ValueError(
                f"Unsupported OneVision task_gate_mode={mode!r}; "
                f"expected one of {sorted(supported)}"
            )
        input_source = str(
            self.config.get("task_gate_input_source", "full_prompt")
            or "full_prompt"
        ).strip().lower()
        if input_source not in TASK_GATE_INPUT_SOURCES:
            raise ValueError(
                f"Unsupported task_gate_input_source={input_source!r}; expected one of "
                f"{list(TASK_GATE_INPUT_SOURCES)!r}"
            )
        if mode == "latest_unit_score":
            if input_source != "full_prompt":
                raise ValueError(
                    "task_gate_mode=latest_unit_score uses the retrieval query "
                    "vector and therefore requires task_gate_input_source=full_prompt"
                )
            if int(self.config.get("retrieval_recent_frames", 0) or 0) <= 0:
                raise ValueError(
                    "task_gate_mode=latest_unit_score requires "
                    "retrieval_recent_frames > 0"
                )
            threshold = float(
                self.config.get("task_gate_latest_unit_score_threshold", 0.0095)
            )
            if not math.isfinite(threshold):
                raise ValueError(
                    "task_gate_latest_unit_score_threshold must be finite"
                )
        return mode

    def _task_gate_input_device(self) -> torch.device:
        _layers, embed_tokens, _norm, _lm_head = self._get_lm_components()
        return embed_tokens.weight.device

    def _build_task_gate_text_inputs(self, raw_text: str) -> Dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": [{"type": "text", "text": raw_text}]}]
        templated = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            templated,
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = self._task_gate_input_device()
        return {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}

    def _task_gate_language_model(self):
        candidates = [
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise RuntimeError("Could not locate the OneVision language model for task gating")

    def _content_token_indices(
        self,
        text_inputs: Dict[str, torch.Tensor],
        raw_text: str,
    ) -> List[int]:
        input_ids = text_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("OneVision task gate requires rank-2 input_ids")
        templated_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        content_ids = [
            int(value)
            for value in self.tokenizer.encode(str(raw_text), add_special_tokens=False)
        ]
        if not content_ids:
            raise ValueError("OneVision task-gate text tokenized to an empty sequence")
        width = len(content_ids)
        starts = [
            start
            for start in range(len(templated_ids) - width + 1)
            if templated_ids[start : start + width] == content_ids
        ]
        if len(starts) != 1:
            raise RuntimeError(
                "OneVision task gate expected exactly one raw-text span in the "
                f"chat template, found {len(starts)}"
            )
        return list(range(starts[0], starts[0] + width))

    def _anchor_hidden_vector(self, text: str, probe_layer: int) -> torch.Tensor:
        probe_text = str(text).strip()
        if not probe_text:
            raise ValueError("Anchor-Hidden gate text must be non-empty")
        text_inputs = self._build_task_gate_text_inputs(probe_text)
        content_indices = self._content_token_indices(text_inputs, probe_text)
        language_model = self._task_gate_language_model()
        with torch.inference_mode():
            outputs = language_model(
                **text_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)):
            raise RuntimeError("Anchor-Hidden gate did not receive hidden states")
        hidden_index = int(probe_layer) + 1
        if not 0 < hidden_index < len(hidden_states):
            raise ValueError(
                f"task_gate_anchor_hidden_layer={probe_layer} is outside the "
                f"resolved {len(hidden_states) - 1}-layer language model"
            )
        hidden = hidden_states[hidden_index]
        index = torch.tensor(content_indices, device=hidden.device, dtype=torch.long)
        vector = hidden[0].index_select(0, index).float().mean(dim=0)
        return torch.nn.functional.normalize(vector, dim=0).detach().cpu()

    def _anchor_hidden_layer(self) -> int:
        layers, _embed_tokens, _norm, _lm_head = self._get_lm_components()
        layer_count = len(layers)
        configured = self.config.get("task_gate_anchor_hidden_layer")
        layer = int(self.config.get("prune_layer", 1)) - 1 if configured is None else int(configured)
        if not 0 <= layer < layer_count:
            raise ValueError(
                "task_gate_anchor_hidden_layer must be in "
                f"[0, {layer_count - 1}], got {layer}"
            )
        return layer

    def _score_anchor_hidden_gate(
        self,
        prompt: str,
        *,
        input_source: str = "full_prompt",
        source: str = "frame_retrieval",
    ) -> Dict[str, Any]:
        layer = self._anchor_hidden_layer()
        past_text = str(self.config.get("task_gate_past_anchor", "") or "").strip()
        nonpast_text = str(self.config.get("task_gate_nonpast_anchor", "") or "").strip()
        signature = (layer, past_text, nonpast_text)
        if getattr(self, "_task_gate_anchor_signature", None) != signature:
            self._task_gate_anchor_vectors = {
                "past": self._anchor_hidden_vector(past_text, layer),
                "nonpast": self._anchor_hidden_vector(nonpast_text, layer),
            }
            self._task_gate_anchor_signature = signature
        question = self._anchor_hidden_vector(prompt, layer).float()
        past_cosine = float(torch.dot(question, self._task_gate_anchor_vectors["past"].float()).item())
        nonpast_cosine = float(torch.dot(question, self._task_gate_anchor_vectors["nonpast"].float()).item())
        threshold = float(self.config.get("task_gate_anchor_hidden_threshold", 0.0))
        return build_anchor_hidden_decision(
            past_cosine=past_cosine,
            nonpast_cosine=nonpast_cosine,
            threshold=threshold,
            probe_layer=layer,
            source=source,
            input_text=prompt,
            input_source=input_source,
        )

    def _choice_logits_router_prompt(self, prompt: str) -> str:
        return build_query_choice_router_prompt(
            prompt,
            normalize_query_choice_prompt_version(self.config),
        )

    def _choice_token_id(self, label: str) -> int:
        retrieve_id, recent_id = resolve_query_choice_token_ids(self.tokenizer)
        if label == "A":
            return retrieve_id
        if label == "B":
            return recent_id
        raise ValueError(f"Unsupported query-choice label: {label!r}")

    def _score_query_choice_logits_gate(
        self,
        prompt: str,
        *,
        input_source: str = "full_prompt",
        source: str = "frame_retrieval",
    ) -> Dict[str, Any]:
        _resolved_text, input_source = resolve_task_gate_input(
            {"task_gate_input_source": input_source},
            question_text=prompt,
            full_prompt=prompt,
        )
        retrieve_id, recent_id = resolve_query_choice_token_ids(self.tokenizer)
        inputs = self._build_task_gate_text_inputs(self._choice_logits_router_prompt(prompt))
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("query_choice_logits expected rank-3 model logits")
        attention_mask = inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            valid = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
            if valid.numel() == 0:
                raise RuntimeError("query_choice_logits received an empty prompt")
            last_position = int(valid[-1].item())
        else:
            last_position = int(inputs["input_ids"].shape[1]) - 1
        retrieve_logit = float(logits[0, last_position, retrieve_id].float().item())
        recent_logit = float(logits[0, last_position, recent_id].float().item())
        return build_query_choice_decision(
            retrieve_logit=retrieve_logit,
            recent_logit=recent_logit,
            retrieve_token_id=retrieve_id,
            recent_token_id=recent_id,
            choice_tokens=query_choice_token_strings(
                self.tokenizer,
                (retrieve_id, recent_id),
            ),
            router_prompt_version=normalize_query_choice_prompt_version(self.config),
            source=source,
            input_text=prompt,
            input_source=input_source,
            threshold=float(
                self.config.get("task_gate_query_choice_threshold", 0.0)
            ),
        )

    def _score_latest_unit_gate(
        self,
        *,
        query_vec: torch.Tensor | None,
        latest_key_vec: torch.Tensor | None,
        latest_frame_id: int | None,
        source: str = "frame_retrieval",
    ) -> Dict[str, Any]:
        if not isinstance(query_vec, torch.Tensor) or not isinstance(
            latest_key_vec, torch.Tensor
        ):
            raise RuntimeError(
                "latest_unit_score gate requires the retrieval query vector and "
                "latest sampled-frame key vector"
            )
        if latest_frame_id is None or int(latest_frame_id) < 0:
            raise RuntimeError(
                "latest_unit_score gate requires a valid latest sampled-frame id"
            )
        if query_vec.numel() != latest_key_vec.numel():
            raise RuntimeError(
                "latest_unit_score query/key dimensions do not match: "
                f"query={query_vec.numel()}, key={latest_key_vec.numel()}"
            )

        score = float(
            torch.dot(latest_key_vec.float(), query_vec.float()).item()
        )
        if not math.isfinite(score):
            raise RuntimeError("latest_unit_score gate produced a non-finite score")
        threshold = float(
            self.config.get("task_gate_latest_unit_score_threshold", 0.0095)
        )
        retrieval_enabled = bool(score <= threshold)
        source_ids = self.state.get("frame_source_ids")
        latest_timestamp = (
            float(source_ids[int(latest_frame_id)])
            if isinstance(source_ids, list)
            and int(latest_frame_id) < len(source_ids)
            else float(latest_frame_id)
        )
        return {
            "enabled": True,
            "mode": "latest_unit_score",
            "source": source,
            "predicted_task_type": (
                "backward" if retrieval_enabled else "realtime"
            ),
            "selected_policy": (
                "retrieval"
                if retrieval_enabled
                else str(
                    self.config.get(
                        "task_gate_realtime_policy",
                        "recent_only",
                    )
                )
            ),
            "retrieval_enabled": retrieval_enabled,
            "score": score,
            "threshold": threshold,
            "metric": "shallow_qk_cosine",
            "representation": (
                "normalized_latest_sampled_frame_k_dot_normalized_question_q"
            ),
            "query_source": "retrieval_query_vector",
            # OneVision's native memory unit is one sampled frame. Keep the
            # cross-backbone field names so downstream gate analysis can read
            # Qwen3-VL and OneVision decisions through the same schema.
            "latest_unit_frame_id": int(latest_frame_id),
            "latest_unit_sample_index": int(latest_frame_id),
            "latest_unit_timestamp": latest_timestamp,
            "unit_granularity": "onevision_sampled_frame",
            "rule": "enable_retrieval_if_latest_unit_score_lte_threshold",
        }

    def _apply_task_gate_to_frames(
        self,
        selected_frames: Sequence[int],
        full_prompt: str,
        *,
        question_text: str | None = None,
        latest_unit_query_vec: torch.Tensor | None = None,
        latest_unit_key_vec: torch.Tensor | None = None,
        latest_unit_frame_id: int | None = None,
    ) -> List[int]:
        mode = self._task_gate_mode()
        if mode == "off":
            decision: Dict[str, Any] = {
                "enabled": False,
                "mode": "off",
                "retrieval_enabled": True,
                "selected_policy": "retrieval",
            }
        elif mode == "latest_unit_score":
            decision = self._score_latest_unit_gate(
                query_vec=latest_unit_query_vec,
                latest_key_vec=latest_unit_key_vec,
                latest_frame_id=latest_unit_frame_id,
            )
        elif mode == "history_layer_decay":
            decision = self._score_history_layer_decay_gate()
        else:
            gate_text, input_source = resolve_task_gate_input(
                self.config,
                question_text=question_text or "",
                full_prompt=full_prompt,
            )
            if mode == "anchor_hidden":
                decision = self._score_anchor_hidden_gate(
                    gate_text,
                    input_source=input_source,
                )
            else:
                decision = self._score_query_choice_logits_gate(
                    gate_text,
                    input_source=input_source,
                )

        frames = [int(frame) for frame in selected_frames]
        if decision["selected_policy"] == "recent_only":
            frame_spans = self.state.get("frame_spans")
            total_frames = len(frame_spans) if isinstance(frame_spans, list) else 0
            recent_n = min(
                max(0, int(self.config.get("task_gate_realtime_recent_frames", 2))),
                total_frames,
            )
            frames = list(range(total_frames - recent_n, total_frames)) if recent_n > 0 else []
            last_selection = self.state.get("last_selection")
            if not isinstance(last_selection, dict):
                last_selection = {}
            last_selection.update(
                {
                    "keep_idx": frames,
                    "recent_idx": frames,
                    "important_seed_idx": [],
                    "important_expanded_idx": [],
                    "long_cluster_indices": [],
                    "long_cluster_scores": [],
                }
            )
            self.state["last_selection"] = last_selection
        elif self._evidence_retrieval_backend() == "siglip":
            frames, evidence_metadata = self._select_siglip_evidence_frames(
                full_prompt=full_prompt,
                question_text=question_text or "",
            )
            decision["evidence_retrieval"] = evidence_metadata
        score_observation = self.state.get(
            "last_latest_unit_score_observation"
        )
        if isinstance(score_observation, dict) and score_observation:
            decision["latest_unit_score_observation"] = dict(score_observation)
        decision["selected_frame_count"] = len(frames)
        self.state["last_gate_decision"] = decision
        return frames
