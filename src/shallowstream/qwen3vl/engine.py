"""Top-level orchestration for the ShallowStream Qwen3-VL V3 runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.shallowstream.evidence_retrieval import validate_evidence_retrieval_config

from .config import MODEL_NAME, _as_bool, _as_int, _retrieval_expansion_strategy
from .decode import Qwen3VLDecodeMixin
from .frame_memory import SampledVideo
from .model_runtime import Qwen3VLModelRuntimeMixin
from .observation import Qwen3VLObservationMixin
from .prompt_generation import Qwen3VLPromptGenerationMixin
from .selection import Qwen3VLSelectionMixin
from .state import _Qwen3VLStreamSession
from .streaming import Qwen3VLStreamingMixin
from .task_gate import Qwen3VLTaskGateMixin


class Qwen3VLInternalKVEngine(
    Qwen3VLObservationMixin,
    Qwen3VLTaskGateMixin,
    Qwen3VLStreamingMixin,
    Qwen3VLPromptGenerationMixin,
    Qwen3VLModelRuntimeMixin,
    Qwen3VLSelectionMixin,
    Qwen3VLDecodeMixin,
):

    def __init__(self, owner: "EvalShallowStreamQwen3VLV3") -> None:
        self.owner = owner
        self.config = owner.config
        self.processor = owner.processor
        self.model = owner.model
        self.memory = owner.memory
        validate_evidence_retrieval_config(self.config, model_family="qwen3vl")
        self.last_gate_decision: Optional[Dict[str, Any]] = None
        self._rekv_sink_path_stats: Dict[str, int] = {}
        self._task_gate_anchor_k: Optional[Dict[str, torch.Tensor]] = None
        self._task_gate_anchor_signature: Optional[Tuple[int, str, str]] = None
        self._task_gate_anchor_hidden: Optional[Dict[str, torch.Tensor]] = None
        self._task_gate_anchor_hidden_signature: Optional[Tuple[int, str, str]] = None
        self._task_gate_text_model = self._load_task_gate_text_model()
        self._task_gate_replay_decisions = self._load_task_gate_replay_decisions()

    def inference(
        self,
        video_path: str,
        prompt: str,
        sample: Optional[SampledVideo] = None,
        stream_batches: Optional[Sequence[SampledVideo]] = None,
        stream_inputs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        self._validate_runtime_config()
        batch_frames = max(int(self.config.get("streaming_prefill_batch_frames", 0) or 0), 0)
        if batch_frames > 0:
            return self._inference_streaming_prefill(
                video_path,
                prompt,
                batch_frames,
                sample=sample,
                batches=stream_batches,
                batch_inputs=stream_inputs,
            )
        return self._inference_full_prefill(video_path, prompt, sample=sample)

    def _inference_full_prefill(
        self,
        video_path: str,
        prompt: str,
        sample: Optional[SampledVideo] = None,
    ) -> Optional[str]:
        if sample is None:
            sample = self.memory.load_sampled_video(video_path)
        frames = sample.frames
        if not frames:
            raise ValueError(f"No frames decoded from video: {video_path}")

        inputs = self._build_inputs(sample.video, sample.metadata, prompt, video_path)
        layers, rotary_emb, norm, lm_head, embed_tokens = self._language_parts()
        prune_layer = self._resolve_prune_layer(layers)

        input_ids = inputs["input_ids"]
        video_positions = self._video_positions(input_ids)
        if video_positions.numel() == 0:
            raise RuntimeError("No Qwen3 visual placeholder tokens were found in the prompt.")
        frames = self._coalesce_frames_to_video_units(frames, self._video_unit_count(inputs, len(frames)))
        frame_positions = self._split_video_positions(video_positions, len(frames), inputs.get("video_grid_thw"))
        local_window_tokens = self._local_window_token_budget(
            [int(positions.numel()) for positions in frame_positions]
        )
        unit_grid_rows = self._unit_grid_rows(inputs, len(frames))

        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(inputs, language)
        input_embeds = language_inputs["inputs_embeds"]
        prefill_positions = self._normalize_rope_positions(
            language_inputs.get("position_ids"),
            seq_len=int(input_embeds.shape[1]),
            device=input_embeds.device,
        )
        visual_pos_masks = language_inputs.get("visual_pos_masks")
        deepstack_visual_embeds = language_inputs.get("deepstack_visual_embeds")
        deepstack_count = len(deepstack_visual_embeds) if isinstance(deepstack_visual_embeds, list) else 0
        streaming_lower_mask = None
        streaming_attention_groups = None
        if _as_bool(self.config, "streaming_lower_mask"):
            if self._rekv_sink_enabled():
                if not self._uses_flash_attention_2():
                    raise RuntimeError("ReKV sink full prefill requires flash_attention_2")
                streaming_attention_groups = self._build_full_prefill_attention_groups(
                    seq_len=int(input_embeds.shape[1]),
                    frame_positions=frame_positions,
                )
            else:
                streaming_lower_mask = self._build_streaming_local_causal_mask(
                    seq_len=int(input_embeds.shape[1]),
                    frame_positions=frame_positions,
                    video_positions=video_positions,
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
        capture_q_layers = {prune_layer - 1}
        token_vote_enabled = (
            self._retrieval_score_strategy() == "shallow_layer_token_vote"
        )
        if token_vote_enabled:
            capture_q_layers.update(
                range(self._retrieval_vote_layer_start(), prune_layer)
            )
        attention_layer = None
        attention_observation_layers = []
        gate_mode = self._task_gate_mode()
        if gate_mode in {"attention_distribution", "history_layer_decay"}:
            attention_layer = self._task_gate_attention_layer(len(layers))
            attention_observation_layers = (
                list(range(prune_layer))
                if gate_mode == "history_layer_decay"
                else self._task_gate_attention_observation_layers(len(layers))
            )
            capture_q_layers.update(attention_observation_layers)
        with torch.no_grad():
            hidden_after_prune, raw_lower_kv = self._forward_lower_layers_raw(
                hidden_states=input_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=prune_layer,
                past_raw_kv={},
                positions=prefill_positions,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                update_cache=True,
                capture_q_layers=capture_q_layers,
                attention_mask=streaming_lower_mask,
                attention_unit_groups=streaming_attention_groups,
                sink_len=int(video_positions[0].item()),
                local_window_tokens=local_window_tokens,
            )

        lower_layer_ids = list(range(prune_layer))
        frame_states = self._build_frame_states(
            frames=frames,
            frame_positions=frame_positions,
            rope_position_ids=prefill_positions,
            raw_lower_kv=raw_lower_kv,
            query_layer=prune_layer - 1,
            input_embeds=input_embeds,
            unit_grid_rows=unit_grid_rows,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        if gate_mode == "recent_context_sufficiency":
            gate_selection = self._recent_context_gate_selection(frame_states)
            recent_answer = self._answer_or_route_from_recent_context(
                gate_selection,
                prompt,
                source="full_recent_prefill",
            )
            if recent_answer is not None:
                del raw_lower_kv, hidden_after_prune, frame_states
                del language_inputs, input_embeds, inputs
                return recent_answer
        probe_query_vecs = self._task_gate_probe_vectors(
            prompt=prompt,
            layers=layers,
            rotary_emb=rotary_emb,
            query_layer=prune_layer - 1,
            context_raw_kv=raw_lower_kv,
            position_start=int(input_embeds.shape[1]),
            context_sink_len=int(video_positions[0].item()),
            context_local_window_tokens=local_window_tokens,
        )
        attention_features = None
        attention_observation_features = None
        token_vote_query_indices = None
        if token_vote_enabled:
            token_vote_query_indices = self._retrieval_vote_query_indices(
                inputs, prompt
            )
        if attention_layer is not None:
            query_indices = self._attention_distribution_query_indices(inputs, prompt)
            attention_observation_features = {}
            for layer_idx in attention_observation_layers:
                _layer, layer_features = self._build_attention_distribution_features(
                    q_entry=raw_lower_kv[layer_idx],
                    k_entry=raw_lower_kv[layer_idx],
                    query_indices=query_indices,
                    frames=frame_states,
                    rotary_emb=rotary_emb,
                    layers=layers,
                    layer_idx=layer_idx,
                )
                attention_observation_features[layer_idx] = layer_features
            attention_features = attention_observation_features[attention_layer]
        selection = self._select_memory(
            frame_states=frame_states,
            raw_lower_kv=raw_lower_kv,
            lower_layer_ids=lower_layer_ids,
            prune_hidden=hidden_after_prune,
            query_layer=prune_layer - 1,
            video_positions=video_positions,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            prompt=prompt,
            probe_query_vecs=probe_query_vecs,
            attention_distribution_layer=attention_layer,
            attention_distribution_features=attention_features,
            attention_distribution_observation_features=attention_observation_features,
            token_vote_query_indices=token_vote_query_indices,
            rotary_emb=rotary_emb,
            layers=layers,
        )
        if self._selected_generate_mode() == "simple_prompt":
            if _as_bool(self.config, "debug"):
                simple_stats = {
                    "selected_generate_mode": "simple_prompt",
                    "selected_short_units": [frame.frame_id for frame in selection.get("short", [])],
                    "selected_short_visual_tokens": sum(
                        int(frame.token_indices.numel()) for frame in selection.get("short", [])
                    ),
                    "recent_units": [frame.frame_id for frame in selection.get("recent", [])],
                    "retrieved_short_units": [frame.frame_id for frame in selection.get("retrieved", [])],
                }
                simple_stats.update(selection.get("token_selection_stats") or {})
                print(f"[{MODEL_NAME}] simple_prompt_selection_stats={json.dumps(simple_stats, ensure_ascii=False)}", flush=True)
            del raw_lower_kv, hidden_after_prune, frame_states, language_inputs, input_embeds, inputs
            with torch.no_grad():
                return self._generate_from_selected_prompt(selection, prompt)
        (
            raw_lower_kv,
            hidden_after_prune,
            visual_pos_masks,
            deepstack_visual_embeds,
            selection,
        ) = self._evict_clustered_long_frame_state(
            selection=selection,
            raw_lower_kv=raw_lower_kv,
            prune_hidden=hidden_after_prune,
            input_seq_len=int(input_ids.shape[1]),
            video_positions=video_positions,
            rope_position_ids=prefill_positions,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        (
            selected_lower_cache,
            selected_hidden,
            selected_positions,
            next_position,
            selected_visual_pos_masks,
            selected_deepstack_visual_embeds,
            stats,
        ) = self._build_selected_context(
            selection=selection,
            raw_lower_kv=raw_lower_kv,
            lower_layer_ids=lower_layer_ids,
            prune_hidden=hidden_after_prune,
            input_seq_len=int(input_ids.shape[1]),
            video_positions=video_positions,
            rope_position_ids=prefill_positions,
            deepstack_count=deepstack_count,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        if _as_bool(self.config, "debug"):
            print(f"[{MODEL_NAME}] internal_kv_stats={json.dumps(stats, ensure_ascii=False)}", flush=True)

        del raw_lower_kv, hidden_after_prune, frame_states, selection, language_inputs, input_embeds, inputs

        with torch.no_grad():
            return self._decode_from_selected_cache(
                selected_lower_cache=selected_lower_cache,
                selected_hidden=selected_hidden,
                selected_positions=selected_positions,
                next_position=next_position,
                selected_visual_pos_masks=selected_visual_pos_masks,
                selected_deepstack_visual_embeds=selected_deepstack_visual_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                norm=norm,
                lm_head=lm_head,
                embed_tokens=embed_tokens,
                prune_layer=prune_layer,
                sink_len=int(stats["prefix_tokens"]),
                local_window_tokens=local_window_tokens,
            )

    def _iter_sampled_video_batches(
        self,
        sample: SampledVideo,
        batch_frames: int,
    ) -> Sequence[SampledVideo]:
        total = len(sample.frames)
        if total <= 0:
            return []
        batch_frames = max(int(batch_frames), 1)
        batches: List[SampledVideo] = []
        for start in range(0, total, batch_frames):
            end = min(start + batch_frames, total)
            frames = list(sample.frames[start:end])
            video = sample.video[start:end].contiguous()
            metadata = dict(sample.metadata)
            metadata["frames_indices"] = [frame.index for frame in frames]
            batches.append(SampledVideo(frames=frames, video=video, metadata=metadata))
        return batches

    def _inference_streaming_prefill(
        self,
        video_path: str,
        prompt: str,
        batch_frames: int,
        sample: Optional[SampledVideo] = None,
        batches: Optional[Sequence[SampledVideo]] = None,
        batch_inputs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        if sample is None:
            sample = self.memory.load_sampled_video(video_path)
        if not sample.frames:
            raise ValueError(f"No frames decoded from video: {video_path}")

        session = self.create_stream_session(video_path)
        if batches is None:
            batches = self._iter_sampled_video_batches(sample, batch_frames)
        if batch_inputs is not None and len(batch_inputs) != len(batches):
            raise ValueError(
                "Prepared stream input count does not match sampled-video batch count: "
                f"{len(batch_inputs)} != {len(batches)}"
            )
        for index, batch in enumerate(batches):
            inputs = batch_inputs[index] if batch_inputs is not None else None
            self._append_stream_frames(session, batch, video_path, inputs=inputs)

        if not session.raw_lower_kv and not session.clusters:
            raise ValueError(f"No streaming frames are available for video: {video_path}")

        if _as_bool(self.config, "debug"):
            stats = {
                "streaming_prefill_batch_frames": int(batch_frames),
                "streaming_prefill_batches": len(batches),
                "streaming_prefill_detail_frames": len(session.frame_states),
                "streaming_prefill_cluster_count": len(session.clusters),
                "streaming_prefill_cluster_frame_counts": [cluster.count for cluster in session.clusters],
                "streaming_prefill_last_timestamp": float(session.last_timestamp),
            }
            print(f"[{MODEL_NAME}] streaming_prefill_stats={json.dumps(stats, ensure_ascii=False)}", flush=True)

        with torch.no_grad():
            return self._answer_stream_question(session, prompt)

    def create_stream_session(self, video_key: str) -> _Qwen3VLStreamSession:
        return _Qwen3VLStreamSession(video_key=str(video_key))

    def stream_inference(
        self,
        session: _Qwen3VLStreamSession,
        video_path: str,
        prompt: str,
    ) -> Optional[str]:
        sample = self.memory.load_sampled_video_since(video_path, session.last_timestamp)
        if sample.frames:
            self._append_stream_frames(session, sample, video_path)
        if not session.raw_lower_kv and not session.clusters:
            raise ValueError(f"No streaming frames are available for video: {video_path}")
        with torch.no_grad():
            return self._answer_stream_question(session, prompt)

    def stream_inference_sample(
        self,
        session: _Qwen3VLStreamSession,
        sample: Optional[SampledVideo],
        video_key: str,
        prompt: str,
    ) -> Optional[str]:
        """Append one prepared incremental chunk, or answer without new video."""

        if sample is not None and sample.frames:
            self._append_stream_frames(session, sample, str(video_key))
        if not session.raw_lower_kv and not session.clusters:
            raise ValueError(
                "Cannot answer a streaming question before any video frames were appended: "
                f"{video_key}"
            )
        with torch.no_grad():
            return self._answer_stream_question(session, prompt)

    def _make_scalar_positions(self, start: int, length: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(int(start), int(start) + int(length), device=device, dtype=torch.long)
        return base.view(1, -1).expand(3, -1).contiguous()

    def _validate_runtime_config(self) -> None:
        batch_frames = int(self.config.get("streaming_prefill_batch_frames", 0) or 0)
        temporal_patch = _as_int(self.config, "video_temporal_patch_size")
        if batch_frames < 0:
            raise ValueError("streaming_prefill_batch_frames must be non-negative")
        if temporal_patch <= 0:
            raise ValueError("video_temporal_patch_size must be positive")
        if batch_frames > 0 and batch_frames % temporal_patch != 0:
            raise ValueError(
                "streaming_prefill_batch_frames must be a multiple of "
                f"video_temporal_patch_size: {batch_frames} % {temporal_patch} != 0"
            )

        mode = self._selected_generate_mode()
        granularity = self._retrieval_selection_granularity()
        score_strategy = self._retrieval_score_strategy()
        self._retrieval_score_reverse()
        expansion_strategy = _retrieval_expansion_strategy(self.config)
        if expansion_strategy == "score_fill" and self._retrieval_expand_next_units() > 0:
            raise ValueError(
                "retrieval_expansion_strategy=score_fill currently requires "
                "retrieval_expand_next_units=0 so recent units cannot overlap "
                "the matched retrieval budget"
            )
        if mode == "simple_prompt" and granularity == "token":
            raise ValueError(
                "retrieval_selection_granularity=token cannot be combined with "
                "selected_generate_mode=simple_prompt because sparse tokens do not "
                "retain a valid Qwen3-VL spatial grid"
            )
        if score_strategy == "shallow_layer_token_vote":
            if self._evidence_retrieval_backend() != "shallow":
                raise ValueError(
                    "retrieval_score_strategy=shallow_layer_token_vote requires "
                    "evidence_retrieval_backend=shallow"
                )
            if granularity != "unit":
                raise ValueError(
                    "retrieval_score_strategy=shallow_layer_token_vote requires "
                    "retrieval_selection_granularity=unit"
                )
            layer_start = self._retrieval_vote_layer_start()
            prune_layer = _as_int(self.config, "prune_layer")
            if layer_start >= prune_layer:
                raise ValueError(
                    "retrieval_vote_layer_start must be smaller than prune_layer: "
                    f"{layer_start} >= {prune_layer}"
                )
            if self._retrieval_vote_topk_tokens_per_layer() <= 0:
                raise ValueError(
                    "retrieval_vote_topk_tokens_per_layer must be positive"
                )
            self._retrieval_vote_query_token_mode()
            if self._retrieval_vote_diversity_pool_multiplier() < 1:
                raise ValueError(
                    "retrieval_vote_diversity_pool_multiplier must be at least 1"
                )
            self._retrieval_vote_diversity_mode()

    def _resolve_prune_layer(self, layers: Sequence[Any]) -> int:
        prune_layer = _as_int(self.config, "prune_layer")
        layer_count = len(layers)
        if not 1 <= prune_layer <= layer_count:
            raise ValueError(
                f"prune_layer must be in [1, {layer_count}], got {prune_layer}"
            )
        return prune_layer

    def _retrieval_recent_units(self) -> int:
        return max(_as_int(self.config, "retrieval_recent_units"), 0)

    def _retrieval_topk_units(self) -> int:
        return max(_as_int(self.config, "retrieval_topk_units"), 0)

    def _retrieval_search_window_units(self) -> int:
        return max(_as_int(self.config, "retrieval_search_last_n_units"), 0)

    def _retrieval_expand_prev_units(self) -> int:
        return max(_as_int(self.config, "retrieval_expand_prev_units"), 0)

    def _retrieval_expand_next_units(self) -> int:
        return max(_as_int(self.config, "retrieval_expand_next_units"), 0)

    def _retrieval_expand_prev_stride_units(self) -> int:
        return max(_as_int(self.config, "retrieval_expand_prev_stride_units"), 1)

    def _retrieval_expand_next_stride_units(self) -> int:
        return max(_as_int(self.config, "retrieval_expand_next_stride_units"), 1)

    def _retrieval_selection_granularity(self) -> str:
        granularity = str(self.config.get("retrieval_selection_granularity", "unit")).strip().lower()
        if granularity in {"frame", "frames", "unit", "units", "temporal_unit", "temporal_units"}:
            return "unit"
        if granularity in {"token", "tokens", "visual_token", "visual_tokens"}:
            return "token"
        raise ValueError(f"Unsupported retrieval_selection_granularity: {granularity!r}")

    def _retrieval_score_strategy(self) -> str:
        strategy = str(
            self.config.get("retrieval_score_strategy", "shallow_unit_cosine")
        ).strip().lower()
        aliases = {
            "shallow": "shallow_unit_cosine",
            "unit_cosine": "shallow_unit_cosine",
            "shallow_unit_cosine": "shallow_unit_cosine",
            "layer_token_vote": "shallow_layer_token_vote",
            "shallow_layer_token_vote": "shallow_layer_token_vote",
        }
        if strategy not in aliases:
            raise ValueError(f"Unsupported retrieval_score_strategy: {strategy!r}")
        return aliases[strategy]

    def _retrieval_vote_layer_start(self) -> int:
        return max(int(self.config.get("retrieval_vote_layer_start", 0) or 0), 0)

    def _retrieval_vote_topk_tokens_per_layer(self) -> int:
        return int(self.config.get("retrieval_vote_topk_tokens_per_layer", 64) or 0)

    def _retrieval_vote_query_token_mode(self) -> str:
        mode = str(
            self.config.get("retrieval_vote_query_token_mode", "all_mean")
            or "all_mean"
        ).strip().lower()
        aliases = {
            "all": "all_mean",
            "mean": "all_mean",
            "all_mean": "all_mean",
            "last": "prompt_last",
            "last_token": "prompt_last",
            "prompt_last": "prompt_last",
        }
        if mode not in aliases:
            raise ValueError(f"Unsupported retrieval_vote_query_token_mode: {mode!r}")
        return aliases[mode]

    def _retrieval_vote_diversity_mode(self) -> str:
        mode = str(
            self.config.get("retrieval_vote_diversity_mode", "off") or "off"
        ).strip().lower()
        aliases = {
            "off": "off",
            "none": "off",
            "divprune": "divprune_maxmin",
            "maxmin": "divprune_maxmin",
            "divprune_maxmin": "divprune_maxmin",
        }
        if mode not in aliases:
            raise ValueError(f"Unsupported retrieval_vote_diversity_mode: {mode!r}")
        return aliases[mode]

    def _retrieval_vote_diversity_pool_multiplier(self) -> int:
        return int(
            self.config.get("retrieval_vote_diversity_pool_multiplier", 4) or 0
        )

    def _selected_generate_mode(self) -> str:
        mode = str(self.config.get("selected_generate_mode", "internal_kv")).strip().lower()
        if mode not in {"simple_prompt", "internal_kv"}:
            raise ValueError(
                "selected_generate_mode must be 'simple_prompt' or 'internal_kv', "
                f"got {mode!r}"
            )
        return mode

    def _retrieval_score_reverse(self) -> bool:
        order = str(self.config.get("retrieval_score_order", "highest")).strip().lower()
        if order not in {"highest", "lowest"}:
            raise ValueError(
                "retrieval_score_order must be 'highest' or 'lowest', "
                f"got {order!r}"
            )
        return order == "highest"
