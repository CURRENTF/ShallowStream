"""Layer-wise observation helpers for the production Qwen3-VL stream path."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import _as_bool, _get_active_attn_implementation
from .frame_memory import SampledVideo
from .memory_profile import (
    append_stream_memory_checkpoint,
    validate_completed_memory_profile,
)
from .state import _Qwen3VLStreamSession


def frame_attention_statistics(
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    *,
    question_indices: torch.Tensor,
    frame_token_spans: Sequence[Tuple[int, int]],
    scaling: float,
    sliding_window: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Reduce exact QK softmax probabilities to one value per image.

    ``q_states`` and ``k_states`` must already include Qwen's Q/K norms and
    rotary embeddings and must have matching attention-head counts.  The
    returned absolute mass keeps text/special-token attention in the
    denominator.  The conditional share renormalizes only across the supplied
    image spans, which makes the six-image recency pattern easier to compare.
    """

    if q_states.ndim != 4 or k_states.ndim != 4:
        raise ValueError("q_states and k_states must be shaped [batch, heads, tokens, dim]")
    if q_states.shape[:2] != k_states.shape[:2] or q_states.shape[-1] != k_states.shape[-1]:
        raise ValueError("q_states and k_states must have matching batch/head/dim shapes")
    if not frame_token_spans:
        raise ValueError("frame_token_spans must contain at least one image span")

    sequence_length = int(k_states.shape[2])
    query_positions = question_indices.to(device=q_states.device, dtype=torch.long).reshape(-1)
    if query_positions.numel() == 0:
        raise ValueError("question_indices must contain at least one token")
    if int(query_positions.min().item()) < 0 or int(query_positions.max().item()) >= sequence_length:
        raise ValueError("question_indices fall outside the prompt sequence")

    previous_end = -1
    for start, end in frame_token_spans:
        start, end = int(start), int(end)
        if not (0 <= start < end <= sequence_length):
            raise ValueError(f"invalid frame token span {(start, end)} for length {sequence_length}")
        if start < previous_end:
            raise ValueError("frame token spans must be ordered and non-overlapping")
        previous_end = end

    selected_q = q_states.index_select(2, query_positions)
    logits = torch.einsum("bhqd,bhkd->bhqk", selected_q.float(), k_states.float())
    logits.mul_(float(scaling))

    key_positions = torch.arange(sequence_length, device=logits.device).view(1, 1, 1, -1)
    absolute_queries = query_positions.view(1, 1, -1, 1)
    visible = key_positions <= absolute_queries
    if sliding_window is not None:
        window = int(sliding_window)
        if window <= 0:
            raise ValueError("sliding_window must be positive when provided")
        visible &= key_positions >= (absolute_queries - window + 1)
    probabilities = torch.softmax(logits.masked_fill(~visible, float("-inf")), dim=-1)

    per_frame = torch.stack(
        [probabilities[..., int(start) : int(end)].sum(dim=-1) for start, end in frame_token_spans],
        dim=-1,
    )
    absolute_mass = per_frame.mean(dim=(0, 1, 2))
    visual_mass = per_frame.sum(dim=-1, keepdim=True)
    conditional = (per_frame / visual_mass.clamp_min(torch.finfo(per_frame.dtype).tiny)).mean(
        dim=(0, 1, 2)
    )
    return {
        "frame_attention_mass": [float(value) for value in absolute_mass.detach().cpu().tolist()],
        "frame_visual_conditional_share": [
            float(value) for value in conditional.detach().cpu().tolist()
        ],
    }


class Qwen3VLObservationMixin:
    @torch.inference_mode()
    def observe_latest_unit_score(
        self,
        session: _Qwen3VLStreamSession,
        prompt: str,
        *,
        layer_number: int,
    ) -> Dict[str, Any]:
        """Compute the production latest-unit gate score without generation.

        This follows the query-prefill portion of ``_answer_stream_question``:
        the exact multimodal prompt suffix is forwarded against the streamed
        visual archive through the requested shallow layer, but neither memory
        selection nor answer decoding is entered.
        """

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("latest-unit observation prompt must not be empty")
        layers, rotary_emb, _norm, _lm_head, embed_tokens = self._language_parts()
        prune_layer = self._resolve_prune_layer(layers)
        depth = int(layer_number)
        if depth != prune_layer:
            raise ValueError(
                "Qwen3-VL latest-unit observation requires "
                f"layer_number == prune_layer ({prune_layer}), got {depth}"
            )
        if not session.frame_states:
            raise RuntimeError(
                "Qwen3-VL latest-unit observation requires at least one temporal unit"
            )

        _prefix_ids, suffix_ids = self._video_prompt_wrapper_token_ids(
            prompt,
            session.video_key,
        )
        if not suffix_ids:
            raise RuntimeError("Qwen3-VL latest-unit prompt suffix is empty")
        embed_device = self._layer_device(embed_tokens, self.owner._input_device())
        question_embeds = self._embed_token_ids(
            suffix_ids,
            embed_tokens,
            embed_device,
        )
        question_positions = self._make_scalar_positions(
            session.next_position,
            int(question_embeds.shape[1]),
            question_embeds.device,
        )
        context_kwargs = self._question_query_context_kwargs(
            question_len=int(question_embeds.shape[1]),
            visual_window_tokens=int(session.local_window_tokens),
            sink_len=int(session.sink_len),
            sink_raw_kv=session.sink_lower_kv,
        )
        query_layer = depth - 1
        _question_hidden, question_cache = self._forward_lower_layers_raw(
            hidden_states=question_embeds,
            layers=layers,
            rotary_emb=rotary_emb,
            start_layer=0,
            end_layer=depth,
            past_raw_kv=session.raw_lower_kv,
            positions=question_positions,
            update_cache=False,
            capture_q_layers={query_layer},
            cache_policy="capture_current",
            **context_kwargs,
        )
        query_vec = self._query_vector_from_current_q(
            question_cache,
            query_layer,
        )
        latest_unit = session.frame_states[-1]
        latest_key = getattr(latest_unit, "key_vec", None)
        if not isinstance(latest_key, torch.Tensor):
            raise RuntimeError("Qwen3-VL latest temporal unit has no normalized K vector")
        score = float(torch.dot(latest_key.float(), query_vec.float()).item())
        if not math.isfinite(score):
            raise RuntimeError("Qwen3-VL latest-unit observation produced a non-finite score")

        result = {
            "metric": "shallow_qk_cosine",
            "representation": (
                "normalized_latest_temporal_unit_k_dot_normalized_question_q"
            ),
            "query_source": "retrieval_query_vector",
            "unit_granularity": "qwen3vl_temporal_patch_unit",
            "unit_source_frame_count": int(
                self.config.get("video_temporal_patch_size", 2)
            ),
            "layer_number": depth,
            "layer_index": query_layer,
            "score": score,
            "latest_unit_timestamp": float(latest_unit.timestamp),
            "sampled_unit_count": int(session.next_frame_id),
            "prompt_token_count": len(suffix_ids),
            "latest_unit_frame_id": int(latest_unit.frame_id),
            "latest_unit_sample_index": int(latest_unit.sample_index),
        }
        del _question_hidden, question_cache, question_embeds, query_vec
        return result

    def _recent_attention_layer_depths(
        self,
        layer_depths: Sequence[object],
    ) -> List[int]:
        layers, _rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        layer_count = len(layers)
        resolved: List[int] = []
        for raw in layer_depths:
            depth = layer_count if str(raw).strip().lower() == "final" else int(raw)
            if not 1 <= depth <= layer_count:
                raise ValueError(
                    f"Observation layer depth {depth} is outside the model's 1..{layer_count} layers"
                )
            if depth not in resolved:
                resolved.append(depth)
        if not resolved:
            raise ValueError("layer_depths must contain at least one layer")
        return resolved

    def _recent_attention_sliding_window(self, layer_idx: int, attn: Any) -> Optional[int]:
        config = getattr(attn, "config", None)
        if config is None:
            config = getattr(getattr(self.model, "config", None), "text_config", None)
        layer_types = getattr(config, "layer_types", None)
        if isinstance(layer_types, (list, tuple)) and int(layer_idx) < len(layer_types):
            if str(layer_types[int(layer_idx)]) != "sliding_attention":
                return None
        else:
            return None
        window = getattr(config, "sliding_window", None)
        return int(window) if window is not None and int(window) > 0 else None

    def _build_recent_attention_prompt(
        self,
        encoded_images: Sequence[Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]],
        *,
        question: str,
        option_suffix: str,
    ) -> Dict[str, Any]:
        if not encoded_images:
            raise ValueError("Recent-image attention observation requires at least one image")
        question = str(question).strip()
        if not question:
            raise ValueError("Recent-image attention observation requires a non-empty question")

        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model_for_generate()
        _layers, _rotary_emb, _norm, _lm_head, embed_tokens = self._language_parts()
        fallback = self.owner._input_device()
        device = self._layer_device(embed_tokens, fallback)

        input_ids_list: List[int] = [int(self.owner.im_start_id)]
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        vision_parts: List[torch.Tensor] = []
        image_grid_rows: List[torch.Tensor] = []
        frame_token_spans: List[Tuple[int, int]] = []
        for visual_embeds, grid_thw, _deepstack in encoded_images:
            embeds = visual_embeds.detach()
            if embeds.dim() == 3 and int(embeds.shape[0]) == 1:
                embeds = embeds[0]
            if embeds.dim() != 2:
                raise RuntimeError(f"recent image visual embeds must be 2D, got {tuple(embeds.shape)}")
            grid = grid_thw.detach().cpu().long().view(3)
            expected_tokens = self._visual_token_count_from_grid(grid)
            if expected_tokens != int(embeds.shape[0]):
                raise RuntimeError(
                    "recent image token/grid mismatch: "
                    f"embeds={int(embeds.shape[0])}, grid_tokens={expected_tokens}"
                )
            input_ids_list.append(int(self.owner.vision_start_id))
            frame_start = len(input_ids_list)
            input_ids_list.extend([int(self.owner.image_token_id)] * int(embeds.shape[0]))
            frame_token_spans.append((frame_start, len(input_ids_list)))
            input_ids_list.append(int(self.owner.vision_end_id))
            vision_parts.append(embeds)
            image_grid_rows.append(grid)

        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        query_start = len(input_ids_list)
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        if not question_ids:
            raise RuntimeError("Tokenizer produced no question tokens")
        input_ids_list.extend(question_ids)
        if option_suffix:
            input_ids_list.extend(tokenizer.encode(option_suffix, add_special_tokens=False))
        query_indices = list(range(query_start, len(input_ids_list)))
        input_ids_list.append(int(self.owner.im_end_id))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = torch.cat(vision_parts, dim=0).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        image_mask = input_ids == int(self.owner.image_token_id)
        if int(image_mask.sum().item()) != int(vision_embeds.shape[0]):
            raise RuntimeError("Recent-image prompt placeholder count does not match visual embeddings")
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            vision_embeds,
        )

        image_grid_thw = torch.stack(image_grid_rows, dim=0).to(device=device, dtype=torch.long)
        try:
            position_ids, _ = text_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        except TypeError:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
            mm_token_type_ids = mm_token_type_ids.masked_fill(image_mask, 1)
            position_ids, _ = text_model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "query_indices": torch.tensor(query_indices, dtype=torch.long),
            "frame_token_spans": frame_token_spans,
        }

    @torch.inference_mode()
    def observe_recent_image_attention(
        self,
        images: Sequence[Any],
        *,
        question: str,
        text_conditions: Mapping[str, str],
        layer_depths: Sequence[object],
    ) -> Dict[str, Any]:
        """Run image-first, no-generation prompts and record question-to-image attention."""

        if not images:
            raise ValueError("observe_recent_image_attention requires recent images")
        resolved_depths = self._recent_attention_layer_depths(layer_depths)
        layers, rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        encoded_images = self._encode_images_for_selected_prompt(images)
        if len(encoded_images) != len(images):
            raise RuntimeError("Vision encoder did not return one embedding group per recent image")

        condition_results: Dict[str, Any] = {}
        for condition_name, option_suffix in text_conditions.items():
            prepared = self._build_recent_attention_prompt(
                encoded_images,
                question=question,
                option_suffix=str(option_suffix),
            )
            captures: Dict[int, Dict[str, torch.Tensor]] = {
                depth - 1: {} for depth in resolved_depths
            }
            handles = []

            def capture(layer_idx: int, name: str):
                def hook(_module, _inputs, output):
                    if not isinstance(output, torch.Tensor):
                        raise RuntimeError(f"Layer {layer_idx + 1} {name}_proj returned a non-tensor")
                    captures[layer_idx][name] = output.detach()
                return hook

            for depth in resolved_depths:
                layer_idx = depth - 1
                attn = self._attention_module(layers[layer_idx])
                handles.append(attn.q_proj.register_forward_hook(capture(layer_idx, "q")))
                handles.append(attn.k_proj.register_forward_hook(capture(layer_idx, "k")))

            language = self._find_language_module()
            try:
                language_output = language(
                    inputs_embeds=prepared["inputs_embeds"],
                    attention_mask=prepared["attention_mask"],
                    position_ids=prepared["position_ids"],
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()

            per_layer: Dict[str, Any] = {}
            for depth in resolved_depths:
                layer_idx = depth - 1
                attn = self._attention_module(layers[layer_idx])
                projected_q = captures[layer_idx].get("q")
                projected_k = captures[layer_idx].get("k")
                if not isinstance(projected_q, torch.Tensor) or not isinstance(projected_k, torch.Tensor):
                    raise RuntimeError(f"Missing Q/K projection capture at Layer {depth}")
                batch, sequence_length, _ = projected_q.shape
                head_dim = int(getattr(attn, "head_dim"))
                q_raw = projected_q.view(batch, sequence_length, -1, head_dim)
                k_raw = projected_k.view(batch, sequence_length, -1, head_dim)
                if hasattr(attn, "q_norm"):
                    q_raw = attn.q_norm(q_raw)
                if hasattr(attn, "k_norm"):
                    k_raw = attn.k_norm(k_raw)
                q_raw = q_raw.transpose(1, 2).contiguous()
                k_raw = k_raw.transpose(1, 2).contiguous()
                layer_positions = prepared["position_ids"].to(device=q_raw.device, dtype=torch.long)
                q_rot = self._apply_rope_to_key(q_raw, layer_positions, rotary_emb)
                k_rot = self._apply_rope_to_key(k_raw, layer_positions, rotary_emb)
                if int(q_rot.shape[1]) % int(k_rot.shape[1]) != 0:
                    raise RuntimeError(f"Layer {depth} Q heads are not divisible by KV heads")
                k_rot = self._repeat_kv(k_rot, int(q_rot.shape[1]) // int(k_rot.shape[1]))
                stats = frame_attention_statistics(
                    q_rot,
                    k_rot,
                    question_indices=prepared["query_indices"],
                    frame_token_spans=prepared["frame_token_spans"],
                    scaling=float(getattr(attn, "scaling", head_dim ** -0.5)),
                    sliding_window=self._recent_attention_sliding_window(layer_idx, attn),
                )
                per_layer[str(depth)] = {
                    **stats,
                    "frame_token_counts": [end - start for start, end in prepared["frame_token_spans"]],
                    "query_token_count": int(prepared["query_indices"].numel()),
                    "query_pooling": (
                        "question_and_option_tokens" if option_suffix else "question_tokens"
                    ),
                    "sequence_token_count": int(sequence_length),
                    "sliding_window": self._recent_attention_sliding_window(layer_idx, attn),
                }
            condition_results[str(condition_name)] = per_layer
            del language_output, prepared, captures

        return {
            "layer_depths": resolved_depths,
            "model_layer_count": len(layers),
            "recent_frame_count": len(images),
            "text_conditions": condition_results,
            "selected_prompt_use_deepstack": False,
            "generation_performed": False,
        }

    def _sample_video_duration_seconds(self, sample: SampledVideo) -> float:
        frames = list(sample.frames or [])
        if not frames:
            return 0.0
        raw_fps = max(
            float(sample.metadata.get("fps", self.config.get("sample_fps", 1.0)) or 1.0),
            1e-6,
        )
        frame_indices = sample.metadata.get("frames_indices")
        if isinstance(frame_indices, torch.Tensor):
            frame_indices = frame_indices.detach().cpu().reshape(-1).tolist()
        elif frame_indices is not None and not isinstance(frame_indices, (list, tuple)):
            try:
                frame_indices = list(frame_indices)
            except TypeError:
                frame_indices = None
        if not frame_indices:
            frame_indices = [int(frame.index) for frame in frames]
        else:
            frame_indices = [int(idx) for idx in frame_indices]

        max_index = max(frame_indices) if frame_indices else int(frames[-1].index)
        max_timestamp = max(float(frame.timestamp) for frame in frames)
        source_frame_dt = 1.0 / raw_fps
        candidates = [
            (float(max_index) + 1.0) / raw_fps,
            max_timestamp + source_frame_dt,
        ]
        metadata_duration = sample.metadata.get("duration")
        if metadata_duration is not None and math.isfinite(float(metadata_duration)):
            candidates.append(float(metadata_duration))
        total_num_frames = sample.metadata.get("total_num_frames")
        if total_num_frames is not None and int(total_num_frames) > 0:
            candidates.append(float(total_num_frames) / raw_fps)
        return max(candidates)

    def prefill_stream_observation(
        self,
        sample: SampledVideo,
        video_path: str,
        batch_frames: int,
        memory_profile: Dict[str, Any] | None = None,
    ) -> _Qwen3VLStreamSession:
        """Run the production streaming lower path without decoding an answer."""

        session = self.create_stream_session(video_path)
        processed_sampled_frames = 0
        checkpoint_frames = set((memory_profile or {}).get("checkpoints_frames") or [])
        for batch in self._iter_sampled_video_batches(sample, batch_frames):
            self._append_stream_frames(session, batch, video_path)
            processed_sampled_frames += len(batch.frames)
            if processed_sampled_frames in checkpoint_frames:
                append_stream_memory_checkpoint(
                    memory_profile,
                    sampled_frames=processed_sampled_frames,
                    session=session,
                )
            if _as_bool(self.config, "full_kv_mode") and torch.cuda.is_available():
                # FullKV preserves every live cache tensor. Only release inactive
                # temporary allocator blocks between mathematically equivalent
                # streaming-prefill batches.
                torch.cuda.empty_cache()
        if not session.raw_lower_kv or not session.frame_states:
            raise RuntimeError("Streaming observation produced no archived lower KV or temporal units")
        if memory_profile:
            validate_completed_memory_profile(memory_profile)
        return session

    def observe_stream_query_layers(
        self,
        session: _Qwen3VLStreamSession,
        prompt: str,
        layer_depths: Sequence[int],
    ) -> Dict[str, Any]:
        """Score every archived temporal unit through the production query path."""

        depths = sorted({int(depth) for depth in layer_depths})
        if not depths or depths[0] < 1:
            raise ValueError("layer_depths must contain positive one-based depths")
        layers, rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        max_depth = max(depths)
        if max_depth > len(layers):
            raise ValueError(f"Requested observation depth {max_depth} exceeds model depth {len(layers)}")

        text_inputs = self._build_text_inputs(prompt)
        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(text_inputs, language)
        question_embeds = language_inputs["inputs_embeds"]
        question_positions = self._make_scalar_positions(
            session.next_position,
            int(question_embeds.shape[1]),
            question_embeds.device,
        )
        capture_layers = [depth - 1 for depth in depths]
        archive_lengths_before = {
            int(layer_idx): int(entry["k"].shape[2])
            for layer_idx, entry in session.raw_lower_kv.items()
        }
        with torch.no_grad():
            question_hidden, question_cache = self._forward_lower_layers_raw(
                hidden_states=question_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=max_depth,
                past_raw_kv=session.raw_lower_kv,
                positions=question_positions,
                update_cache=False,
                capture_q_layers=capture_layers,
                cache_policy="capture_current",
                sink_len=int(session.sink_len),
                sink_raw_kv=session.sink_lower_kv,
                local_window_tokens=int(session.local_window_tokens),
            )

        scores: Dict[int, List[float]] = {}
        for depth in depths:
            layer_idx = depth - 1
            archived_key, _ = self._raw_layer(session.raw_lower_kv, layer_idx)
            current = question_cache.get(layer_idx)
            if not isinstance(current, dict) or not isinstance(current.get("q"), torch.Tensor):
                raise RuntimeError(f"Missing current query Q at Layer {depth}")
            query = current["q"]
            query_indices = torch.arange(int(query.shape[2]), device=query.device, dtype=torch.long)
            query_vec = self._normalized_query_vector(
                query,
                query_indices,
                key_head_count=int(archived_key.shape[1]),
            )
            layer_scores: List[float] = []
            for frame in session.frame_states:
                frame_vec = self._normalized_key_vector(archived_key, frame.token_indices)
                layer_scores.append(float(torch.dot(frame_vec.float(), query_vec.float()).item()))
            scores[depth] = layer_scores

        archive_lengths_after = {
            int(layer_idx): int(entry["k"].shape[2])
            for layer_idx, entry in session.raw_lower_kv.items()
        }
        if archive_lengths_after != archive_lengths_before:
            raise RuntimeError("Query observation mutated the exact video KV archive")
        active_lengths = {
            int(layer_idx): int(entry["k"].shape[2])
            for layer_idx, entry in session.active_lower_kv.items()
        }
        video_backend = _get_active_attn_implementation(self.model)
        if _as_bool(self.config, "streaming_lower_mask") and video_backend == "flash_attention_2":
            video_backend = "flash_attention_2_physical_sink_local"
        result = {
            "scores": scores,
            "unit_count": len(session.frame_states),
            "question_token_count": int(question_embeds.shape[1]),
            "archive_token_count_by_layer": archive_lengths_after,
            "active_token_count_by_layer": active_lengths,
            "shallow_prefill_local_window_frames": self._local_window_frames(),
            "local_window_units": self._local_window_units(),
            "local_window_tokens": int(session.local_window_tokens),
            "use_rekv_sink": _as_bool(self.config, "use_rekv_sink"),
            "rekv_sink_len": int(session.sink_len),
            "rekv_sink_path_stats": dict(getattr(self, "_rekv_sink_path_stats", {})),
            "retrieval_search_last_n_units": self._retrieval_search_window_units(),
            "video_lower_attention_backend": video_backend,
            "query_lower_attention_backend": _get_active_attn_implementation(self.model),
            "query_history_scope": "global_exact_archive",
            "query_pooling_scope": "full_text_prompt_chat_template",
            "archive_device": str(self._streaming_archive_device()),
        }
        del question_hidden, question_cache, question_embeds, language_inputs, text_inputs
        return result
