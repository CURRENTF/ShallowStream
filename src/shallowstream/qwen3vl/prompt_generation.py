"""Selected visual-prompt reconstruction and generation for ShallowStream Qwen3-VL V3."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from src.shallowstream.task_gate import (
    build_recent_context_sufficiency_prompt,
    resolve_query_choice_token_ids,
)

from .config import MODEL_NAME, _as_bool, _as_int, _full_kv_enabled
from .state import _FrameKVState, _LongKVCluster, _RecentSourceFrame


_PromptItem = Tuple[str, torch.Tensor, torch.Tensor, str, Optional[List[torch.Tensor]]]


def _sync_cuda_if_requested(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _first_token_recorder(sync_cuda: bool):
    from transformers import StoppingCriteria

    class _FirstTokenRecorder(StoppingCriteria):
        def __init__(self) -> None:
            super().__init__()
            self.first_token_time: Optional[float] = None

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self.first_token_time is None:
                _sync_cuda_if_requested(sync_cuda)
                self.first_token_time = time.perf_counter()
            return False

    return _FirstTokenRecorder()


class Qwen3VLPromptGenerationMixin:
    def _recent_context_gate_selection(
        self,
        frames: Sequence[_FrameKVState],
    ) -> Dict[str, Any]:
        recent_units = int(
            self.config.get("task_gate_recent_sufficiency_units", 2) or 0
        )
        recent = list(frames[-recent_units:]) if recent_units > 0 else []
        return {
            "query_vec": None,
            "short": recent,
            "clusters": [],
            "all_clusters": [],
            "retrieved": [],
            "retrieved_seed": [],
            "recent": recent,
            "search_candidates": [],
            "short_scores": [],
            "token_selection_stats": {
                "retrieval_selection_granularity": "none",
                "memory_policy": "recent_context_sufficiency",
            },
            "probe_query_vecs": {},
        }

    def _selected_prompt_position_plan(
        self,
        position_ids: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        # Historical/relative mode intentionally follows normal HF generate:
        # Qwen replaces the supplied prefill positions with its scalar
        # cache-position sequence. Absolute mode keeps the explicit MRoPE hook.
        if self._rope_position_mode() == "relative":
            return {"position_ids": position_ids}, None
        return {}, position_ids

    def _question_query_context_kwargs(
        self,
        *,
        question_len: int,
        visual_window_tokens: int,
        sink_len: int,
        sink_raw_kv: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, Any]:
        # The released no-sink path uses FlashAttention's normal causal mask
        # over the retained visual KV. Supplying an equivalent-looking explicit
        # mask switches kernels and measurably changes the retrieval query.
        if not self._rekv_sink_enabled():
            return {}
        question_len = int(question_len)
        visual_window_tokens = int(visual_window_tokens)
        if question_len <= 0:
            raise ValueError("question_len must be positive")
        if visual_window_tokens <= 0:
            raise ValueError("visual_window_tokens must be positive")
        return {
            "sink_len": int(sink_len),
            "sink_raw_kv": sink_raw_kv,
            # Include the whole question tail in the FlashAttention window so
            # every question token retains the configured visual window.
            "local_window_tokens": visual_window_tokens + question_len,
        }

    def _compose_reused_stream_shallow_context(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_lower_cache: Dict[int, Dict[str, torch.Tensor]],
        prefix_positions: torch.Tensor,
        selected_visual_cache: Dict[int, Dict[str, torch.Tensor]],
        selected_visual_hidden: torch.Tensor,
        selected_visual_positions: torch.Tensor,
        selected_visual_mask: Optional[torch.Tensor],
        question_hidden: torch.Tensor,
        question_query_cache: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compact selected shallow state without recomputing query tokens."""
        prefix_len = int(prefix_hidden.shape[1])
        visual_len = int(selected_visual_hidden.shape[1])
        question_len = int(question_hidden.shape[1])
        if question_len <= 0:
            raise RuntimeError("Qwen3-VL internal_kv requires a non-empty prefetched question")
        if int(prefix_positions.shape[-1]) != prefix_len:
            raise RuntimeError("Qwen3-VL prefix hidden/position lengths do not match")

        selected_visual_cache, selected_visual_positions = self._offset_stream_visual_context(
            visual_cache=selected_visual_cache,
            visual_positions=selected_visual_positions,
            prefix_len=prefix_len,
            lower_layer_ids=lower_layer_ids,
        )
        question_positions = self._make_scalar_positions(
            prefix_len + visual_len,
            question_len,
            question_hidden.device,
        )
        reused_question_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            entry = question_query_cache.get(int(layer_idx))
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Missing prefetched Qwen3-VL question cache at layer {int(layer_idx)}"
                )
            k = entry.get("k")
            v = entry.get("v")
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise RuntimeError(
                    f"Invalid prefetched Qwen3-VL question cache at layer {int(layer_idx)}"
                )
            if int(k.shape[2]) != question_len or int(v.shape[2]) != question_len:
                raise RuntimeError(
                    "Qwen3-VL prefetched question KV length does not match its cutoff hidden state"
                )
            reused_question_cache[int(layer_idx)] = {
                "k": k,
                "v": v,
                # K is archived before RoPE, so compact positions can be
                # assigned without another shallow forward.
                "positions": question_positions.to(device=k.device, dtype=torch.long),
            }

        if prefix_len > 0:
            selected_lower_cache = self._concat_lower_caches(
                prefix_lower_cache,
                selected_visual_cache,
                lower_layer_ids,
            )
        else:
            selected_lower_cache = selected_visual_cache
        selected_lower_cache = self._concat_lower_caches(
            selected_lower_cache,
            reused_question_cache,
            lower_layer_ids,
        )

        selected_hidden = torch.cat(
            [
                prefix_hidden,
                selected_visual_hidden.to(device=prefix_hidden.device, dtype=prefix_hidden.dtype),
                question_hidden.to(device=prefix_hidden.device, dtype=prefix_hidden.dtype),
            ],
            dim=1,
        )
        selected_positions = torch.cat(
            [
                prefix_positions.to(device=selected_hidden.device, dtype=torch.long),
                selected_visual_positions.to(device=selected_hidden.device, dtype=torch.long),
                question_positions.to(device=selected_hidden.device, dtype=torch.long),
            ],
            dim=-1,
        )
        if selected_visual_mask is None:
            selected_visual_mask = torch.ones(
                (1, visual_len),
                device=selected_hidden.device,
                dtype=torch.bool,
            )
        selected_masks = torch.cat(
            [
                torch.zeros((1, prefix_len), device=selected_hidden.device, dtype=torch.bool),
                selected_visual_mask.to(selected_hidden.device),
                torch.zeros((1, question_len), device=selected_hidden.device, dtype=torch.bool),
            ],
            dim=1,
        )
        return selected_lower_cache, selected_hidden, selected_positions, selected_masks

    def _generate_selected_prompt_ids(
        self,
        generate_kwargs: Dict[str, Any],
        *,
        prompt_len: int,
        outputs_include_prompt: bool = True,
        language_position_ids: Optional[torch.Tensor] = None,
        position_prompt_len: Optional[int] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        from transformers import StoppingCriteriaList

        max_new_tokens = _as_int(self.config, "max_new_tokens")
        force_exact = _as_bool(self.config, "force_exact_new_tokens")
        sync_cuda = _as_bool(self.config, "latency_sync_cuda")
        recorder = _first_token_recorder(sync_cuda)
        kwargs = dict(generate_kwargs)
        kwargs["max_new_tokens"] = max_new_tokens
        # Qwen3-VL supports keeping only the last prompt position's vocabulary
        # logits. This also prevents AnyRes image prompts from materializing a
        # multi-gigabyte full-sequence logits tensor.
        kwargs["logits_to_keep"] = 1
        if force_exact:
            kwargs["min_new_tokens"] = max_new_tokens
        kwargs["stopping_criteria"] = StoppingCriteriaList([recorder])

        hook = None
        if language_position_ids is not None:
            language = self._find_language_module()
            base_positions = language_position_ids.detach()
            position_prompt_len = int(position_prompt_len or prompt_len)
            if position_prompt_len < prompt_len:
                raise ValueError("position_prompt_len must cover the supplied prompt tokens")
            if base_positions.dim() == 2:
                base_positions = base_positions.unsqueeze(1)
            if base_positions.dim() != 3 or int(base_positions.shape[0]) != 3:
                raise RuntimeError(
                    f"selected prompt position_ids must be (3, 1, L), got {tuple(base_positions.shape)}"
                )

            def _inject_selected_prompt_state(_module, args, call_kwargs):
                if not isinstance(call_kwargs, dict):
                    raise RuntimeError("Qwen3 selected-prompt generation requires keyword model inputs")
                cache_position = call_kwargs.get("cache_position")
                is_prefill = (
                    cache_position is None
                    or int(cache_position.reshape(-1)[0].item()) == 0
                )
                if is_prefill:
                    call_kwargs["position_ids"] = base_positions.to(
                        device=call_kwargs["inputs_embeds"].device,
                        dtype=torch.long,
                    )
                    call_kwargs["visual_pos_masks"] = visual_pos_masks
                    call_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
                else:
                    delta = cache_position.to(device=base_positions.device, dtype=torch.long).reshape(-1)
                    delta = delta - (position_prompt_len - 1)
                    call_kwargs["position_ids"] = (
                        base_positions[..., -1:] + delta.view(1, 1, -1)
                    ).to(device=call_kwargs["inputs_embeds"].device, dtype=torch.long)
                    call_kwargs["visual_pos_masks"] = None
                    call_kwargs["deepstack_visual_embeds"] = None
                return args, call_kwargs

            hook = language.register_forward_pre_hook(_inject_selected_prompt_state, with_kwargs=True)

        _sync_cuda_if_requested(sync_cuda)
        generation_start_time = time.perf_counter()
        try:
            generated_ids = self.model.generate(**kwargs)
        finally:
            if hook is not None:
                hook.remove()
        _sync_cuda_if_requested(sync_cuda)
        generation_end_time = time.perf_counter()
        generated_count = int(generated_ids.shape[1]) - (int(prompt_len) if outputs_include_prompt else 0)
        if force_exact and generated_count != max_new_tokens:
            raise RuntimeError(
                "ShallowStream Qwen3-VL fixed-token benchmark generated "
                f"{generated_count} tokens, expected {max_new_tokens}"
            )
        self._last_decode_observation = {
            "generation_start_time": generation_start_time,
            "first_token_time": recorder.first_token_time,
            "generation_end_time": generation_end_time,
            "generated_tokens": generated_count,
        }
        return generated_ids

    def _visual_token_count_from_grid(self, grid_thw: torch.Tensor) -> int:
        row = grid_thw.detach().cpu().long().view(3)
        merge_area = max(int(getattr(self.owner, "merge_size", 2) or 2), 1) ** 2
        return max(1, int(row.prod().item()) // merge_area)

    def _get_image_feature_model(self):
        if hasattr(self.model, "get_image_features"):
            return self.model
        multimodal = getattr(self.model, "model", self.model)
        if hasattr(multimodal, "get_image_features"):
            return multimodal
        raise RuntimeError("Could not locate Qwen3-VL get_image_features for recent image re-encoding.")

    def _visual_encoder_device(self) -> torch.device:
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is None:
            visual = getattr(self.model, "visual", None)
        if visual is not None:
            for parameter in visual.parameters():
                return parameter.device
            for buffer in visual.buffers():
                return buffer.device
        return self.owner._input_device()

    def _flatten_vision_features(self, features: Any) -> torch.Tensor:
        if hasattr(features, "pooler_output"):
            return self._flatten_vision_features(features.pooler_output)
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, (tuple, list)):
            if features and all(isinstance(item, torch.Tensor) for item in features):
                return torch.cat(list(features), dim=0)
            first = features[0] if features else None
            if isinstance(first, torch.Tensor):
                return first
            if isinstance(first, (tuple, list)) and first and all(isinstance(item, torch.Tensor) for item in first):
                return torch.cat(list(first), dim=0)
        raise TypeError(f"Unexpected Qwen3 vision feature type: {type(features)}")

    @torch.inference_mode()
    def _encode_images_for_selected_prompt(
        self,
        images: Sequence[Image.Image],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]]:
        if not images:
            return []
        content = [{"type": "image", "image": image.convert("RGB")} for image in images]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        device = self._visual_encoder_device()
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        features = self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        deepstack_embeds: Optional[List[torch.Tensor]] = None
        if isinstance(features, (tuple, list)) and len(features) == 2:
            base_features, deepstack_features = features
            image_embeds = self._flatten_vision_features(base_features)
            if isinstance(deepstack_features, (tuple, list)) and all(
                isinstance(item, torch.Tensor) for item in deepstack_features
            ):
                deepstack_embeds = list(deepstack_features)
        else:
            image_embeds = self._flatten_vision_features(features)
        merge_area = max(int(getattr(self.owner, "merge_size", 2) or 2), 1) ** 2
        token_counts = [
            max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
            for row in image_grid_thw
        ]
        expected_tokens = sum(token_counts)
        if expected_tokens != int(image_embeds.shape[0]):
            raise RuntimeError(
                "recent image re-encode token/grid mismatch: "
                f"embeds={int(image_embeds.shape[0])}, grid_tokens={expected_tokens}"
            )

        outputs: List[Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]] = []
        offset = 0
        for token_count, grid in zip(token_counts, image_grid_thw):
            end = offset + int(token_count)
            outputs.append(
                (
                    image_embeds[offset:end].detach().to(device="cpu", dtype=torch.bfloat16),
                    grid.detach().cpu().long(),
                    (
                        [layer[offset:end].detach().to(device="cpu", dtype=torch.bfloat16) for layer in deepstack_embeds]
                        if deepstack_embeds is not None
                        else None
                    ),
                )
            )
            offset = end
        return outputs

    def _cached_video_prompt_items(
        self,
        frames: Sequence[_FrameKVState],
    ) -> List[_PromptItem]:
        items: List[_PromptItem] = []
        for frame in frames:
            if isinstance(frame.visual_embeds, torch.Tensor) and isinstance(frame.grid_thw, torch.Tensor):
                items.append(
                    (
                        "video",
                        frame.visual_embeds.detach(),
                        frame.grid_thw.detach().cpu().long(),
                        self._format_qwen3_timestamp(frame.timestamp),
                        frame.deepstack_embeds,
                    )
                )
        return items

    def _frame_image_prompt_items(
        self,
        frames: Sequence[_FrameKVState],
        *,
        require_images: bool = False,
    ) -> List[_PromptItem]:
        image_jobs: List[Tuple[Image.Image, float]] = []
        expand_source_frames = _as_bool(
            self.config,
            "selected_prompt_expand_recent_source_frames",
        )
        for frame in frames:
            job_count_before = len(image_jobs)
            source_images = (
                frame.source_images
                if expand_source_frames and isinstance(frame.source_images, list)
                else []
            )
            source_timestamps = (
                frame.source_timestamps if isinstance(frame.source_timestamps, list) else []
            )
            if source_images:
                image_jobs.extend(
                    (
                        image,
                        (
                            float(source_timestamps[idx])
                            if idx < len(source_timestamps)
                            else float(frame.timestamp)
                        ),
                    )
                    for idx, image in enumerate(source_images)
                    if isinstance(image, Image.Image)
                )
            elif isinstance(frame.image, Image.Image):
                image_jobs.append((frame.image, float(frame.timestamp)))
            if require_images and len(image_jobs) == job_count_before:
                raise RuntimeError(
                    "Qwen3-VL all-image simple_prompt selected a short-memory unit "
                    f"without a retained source image: frame_id={frame.frame_id}"
                )

        items: List[_PromptItem] = []
        if image_jobs:
            encoded = self._encode_images_for_selected_prompt([image for image, _timestamp in image_jobs])
            if len(encoded) != len(image_jobs):
                raise RuntimeError(
                    "Qwen3-VL image encoder returned an unexpected number of feature groups: "
                    f"encoded={len(encoded)}, requested={len(image_jobs)}"
                )
            for (_image, timestamp), (embeds, grid, deepstack) in zip(image_jobs, encoded):
                items.append(
                    (
                        "image",
                        embeds,
                        grid,
                        self._format_qwen3_timestamp(timestamp),
                        deepstack,
                    )
                )

        if items:
            return items
        if require_images and frames:
            raise RuntimeError(
                "Qwen3-VL all-image simple_prompt could not encode every selected short-memory unit"
            )
        for frame in frames:
            if isinstance(frame.visual_embeds, torch.Tensor) and isinstance(frame.grid_thw, torch.Tensor):
                # Fallback keeps generation alive if a legacy state lacks PIL images.
                items.append((
                    "video",
                    frame.visual_embeds.detach(),
                    frame.grid_thw.detach().cpu().long(),
                    self._format_qwen3_timestamp(frame.timestamp),
                    frame.deepstack_embeds,
                ))
        return items

    def _recent_image_prompt_items(
        self,
        frames: Sequence[_FrameKVState],
    ) -> List[_PromptItem]:
        return self._frame_image_prompt_items(frames)

    def _recent_source_frame_prompt_items(
        self,
        frames: Sequence[_RecentSourceFrame],
    ) -> List[_PromptItem]:
        encoded = self._encode_images_for_selected_prompt(
            [frame.image for frame in frames]
        )
        if len(encoded) != len(frames):
            raise RuntimeError(
                "Qwen3-VL image encoder returned an unexpected number of recent "
                f"source-frame features: encoded={len(encoded)}, requested={len(frames)}"
            )
        return [
            (
                "image",
                embeds,
                grid,
                self._format_qwen3_timestamp(frame.timestamp),
                deepstack,
            )
            for frame, (embeds, grid, deepstack) in zip(frames, encoded)
        ]

    def _cluster_image_prompt_items(
        self,
        clusters: Sequence[_LongKVCluster],
    ) -> List[_PromptItem]:
        image_jobs: List[Tuple[Image.Image, str]] = []
        for cluster in clusters:
            image = cluster.representative_image
            if not isinstance(image, Image.Image):
                raise RuntimeError(
                    "Qwen3-VL all-image simple_prompt selected a long cluster without "
                    f"a representative source image: frames={cluster.start_frame}-{cluster.end_frame}"
                )
            image_jobs.append(
                (
                    image,
                    self._format_qwen3_cluster_timestamp(
                        cluster.start_time,
                        cluster.end_time,
                    ),
                )
            )

        encoded = self._encode_images_for_selected_prompt(
            [image for image, _timestamp in image_jobs]
        )
        if len(encoded) != len(image_jobs):
            raise RuntimeError(
                "Qwen3-VL image encoder returned an unexpected number of long-cluster features: "
                f"encoded={len(encoded)}, requested={len(image_jobs)}"
            )
        return [
            ("image", embeds, grid, timestamp, deepstack)
            for (_image, timestamp), (embeds, grid, deepstack) in zip(image_jobs, encoded)
        ]

    def _selected_prompt_groups(
        self,
        selection: Dict[str, Any],
    ) -> List[Tuple[str, List[_PromptItem]]]:
        groups: List[Tuple[str, List[_PromptItem]]] = []
        reencode_all_as_images = _as_bool(
            self.config,
            "selected_prompt_reencode_all_as_images",
        )

        def _cluster_items(clusters: Sequence[_LongKVCluster]) -> List[_PromptItem]:
            items: List[_PromptItem] = []
            for cluster in clusters:
                if isinstance(cluster.visual_embeds, torch.Tensor) and isinstance(cluster.grid_thw, torch.Tensor):
                    items.append(
                        (
                            "video",
                            cluster.visual_embeds.detach(),
                            cluster.grid_thw.detach().cpu().long(),
                            self._format_qwen3_cluster_timestamp(cluster.start_time, cluster.end_time),
                            cluster.deepstack_embeds,
                        )
                    )
            return items

        selected_clusters = selection.get("clusters", [])
        cluster_items = (
            self._cluster_image_prompt_items(selected_clusters)
            if reencode_all_as_images
            else _cluster_items(selected_clusters)
        )
        if cluster_items:
            groups.append(("Compressed historical video memory:", cluster_items))

        recent_ids = {int(frame.frame_id) for frame in selection.get("recent", [])}
        retrieved_frames = [
            frame
            for frame in selection.get("retrieved", [])
            if int(frame.frame_id) not in recent_ids
        ]
        retrieved_items = (
            self._frame_image_prompt_items(retrieved_frames, require_images=True)
            if reencode_all_as_images
            else self._cached_video_prompt_items(retrieved_frames)
        )
        if retrieved_items:
            groups.append(("Retrieved short video memory:", retrieved_items))

        recent_source_frames = selection.get("recent_source_frames")
        recent_items = (
            self._recent_source_frame_prompt_items(recent_source_frames)
            if isinstance(recent_source_frames, list)
            else self._frame_image_prompt_items(
                selection.get("recent", []),
                require_images=reencode_all_as_images,
            )
        )
        if recent_items:
            groups.append(("Recent video memory:", recent_items))

        return groups

    @torch.inference_mode()
    def _generate_from_selected_prompt(
        self,
        selection: Dict[str, Any],
        prompt: str,
        *,
        prefill_callback: Optional[Callable[..., torch.Tensor]] = None,
    ) -> str:
        groups = self._selected_prompt_groups(selection)
        if not groups:
            inputs = self._build_text_inputs(prompt, include_answer_prefix=True)
            prompt_len = int(inputs["input_ids"].shape[1])
            generated_ids = self._generate_selected_prompt_ids(
                {
                    **inputs,
                    "do_sample": _as_bool(self.config, "do_sample"),
                },
                prompt_len=prompt_len,
            )
            trimmed = [
                generated_ids[0][prompt_len:]
                if int(generated_ids.shape[1]) > prompt_len
                else generated_ids[0]
            ]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model_for_generate()
        device = getattr(self.model, "device", self.owner._input_device())
        device = torch.device(device) if not isinstance(device, torch.device) else device

        input_ids_list: List[int] = []
        input_ids_list.append(int(self.owner.im_start_id))
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))

        vision_parts: List[torch.Tensor] = []
        deepstack_parts: Optional[List[List[torch.Tensor]]] = None
        deepstack_complete = True
        image_grid_rows: List[torch.Tensor] = []
        video_grid_rows: List[torch.Tensor] = []
        use_labels = (
            _as_bool(self.config, "selected_prompt_use_labels")
            if "selected_prompt_use_labels" in self.config
            else False
        )
        use_timestamps = _as_bool(self.config, "selected_prompt_use_timestamps")
        for label, items in groups:
            if use_labels:
                input_ids_list.extend(
                    tokenizer.encode(f"{label}\n", add_special_tokens=False)
                )
            for modality, visual_embeds, grid_thw, timestamp_text, item_deepstack in items:
                embeds_2d = visual_embeds.detach()
                if embeds_2d.dim() == 3 and int(embeds_2d.shape[0]) == 1:
                    embeds_2d = embeds_2d[0]
                if embeds_2d.dim() != 2:
                    raise RuntimeError(f"selected visual embeds must be 2D, got {tuple(embeds_2d.shape)}")
                grid = grid_thw.detach().cpu().long().view(3)
                expected_tokens = self._visual_token_count_from_grid(grid)
                if expected_tokens != int(embeds_2d.shape[0]):
                    raise RuntimeError(
                        "selected visual token/grid mismatch: "
                        f"embeds={int(embeds_2d.shape[0])}, grid_tokens={expected_tokens}, grid={grid.tolist()}"
                    )
                if use_timestamps:
                    input_ids_list.extend(tokenizer.encode(f"{timestamp_text}\n", add_special_tokens=False))
                input_ids_list.append(int(self.owner.vision_start_id))
                if modality == "image":
                    input_ids_list.extend([int(self.owner.image_token_id)] * int(embeds_2d.shape[0]))
                    image_grid_rows.append(grid)
                elif modality == "video":
                    input_ids_list.extend([int(self.owner.video_token_id)] * int(embeds_2d.shape[0]))
                    video_grid_rows.append(grid)
                else:
                    raise RuntimeError(f"Unsupported selected prompt modality: {modality!r}")
                input_ids_list.append(int(self.owner.vision_end_id))
                vision_parts.append(embeds_2d)
                if item_deepstack is None:
                    deepstack_complete = False
                else:
                    if deepstack_parts is None:
                        deepstack_parts = [[] for _ in item_deepstack]
                    if len(item_deepstack) != len(deepstack_parts):
                        raise RuntimeError("selected prompt items have inconsistent DeepStack layer counts")
                    for layer_idx, layer_embeds in enumerate(item_deepstack):
                        if int(layer_embeds.shape[0]) != int(embeds_2d.shape[0]):
                            raise RuntimeError(
                                "selected prompt DeepStack/token mismatch: "
                                f"layer={layer_idx}, deepstack={int(layer_embeds.shape[0])}, "
                                f"visual={int(embeds_2d.shape[0])}"
                            )
                        deepstack_parts[layer_idx].append(layer_embeds.detach())
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))

        input_ids_list.extend(tokenizer.encode(prompt, add_special_tokens=False))
        input_ids_list.append(int(self.owner.im_end_id))
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.append(int(self.owner.im_start_id))
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
        answer_prefix = getattr(self.owner, "assistant_answer_prefix", None)
        if answer_prefix is not None and not callable(answer_prefix):
            raise RuntimeError("owner.assistant_answer_prefix must be callable")
        input_ids_list.extend(
            tokenizer.encode(
                str(answer_prefix() or "") if answer_prefix is not None else "",
                add_special_tokens=False,
            )
        )

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = torch.cat(vision_parts, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = input_ids == int(self.owner.image_token_id)
        video_mask = input_ids == int(self.owner.video_token_id)
        visual_mask = image_mask | video_mask
        visual_mask_expanded = visual_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if int(visual_mask.sum().item()) != int(vision_embeds.shape[0]):
            raise RuntimeError(
                f"selected prompt placeholder mismatch: placeholders={int(visual_mask.sum().item())}, "
                f"embeds={int(vision_embeds.shape[0])}"
            )
        inputs_embeds = inputs_embeds.masked_scatter(visual_mask_expanded, vision_embeds)
        use_deepstack = _as_bool(self.config, "selected_prompt_use_deepstack")
        selected_deepstack = None
        if use_deepstack:
            if not deepstack_complete or deepstack_parts is None:
                raise RuntimeError(
                    "selected simple prompt is missing Qwen3-VL DeepStack features; "
                    "refusing to run a semantically incomplete generation"
                )
            selected_deepstack = [
                torch.cat(parts, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                for parts in deepstack_parts
            ]

        image_grid_thw = (
            torch.stack(image_grid_rows, dim=0).to(device=device, dtype=torch.long)
            if image_grid_rows
            else None
        )
        video_grid_thw = (
            torch.stack(video_grid_rows, dim=0).to(device=device, dtype=torch.long)
            if video_grid_rows
            else None
        )
        try:
            position_ids, _ = text_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        except TypeError:
            # Newer Qwen3-VL implementations require explicit modality ids:
            # text=0, image=1, video=2.
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
            mm_token_type_ids = mm_token_type_ids.masked_fill(image_mask, 1)
            mm_token_type_ids = mm_token_type_ids.masked_fill(video_mask, 2)
            position_ids, _ = text_model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        generate_kwargs: Dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": _as_int(self.config, "max_new_tokens"),
            "do_sample": _as_bool(self.config, "do_sample"),
        }
        position_kwargs, hooked_position_ids = self._selected_prompt_position_plan(position_ids)
        generate_kwargs.update(position_kwargs)
        if _as_bool(self.config, "do_sample"):
            generate_kwargs["temperature"] = max(float(self.config.get("decode_temperature", 1.0)), 1e-6)

        prompt_len = int(input_ids.shape[1])
        generate_fn = prefill_callback or self._generate_selected_prompt_ids
        generated_ids = generate_fn(
            generate_kwargs,
            prompt_len=prompt_len,
            outputs_include_prompt=False,
            language_position_ids=hooked_position_ids,
            visual_pos_masks=visual_mask if use_deepstack else None,
            deepstack_visual_embeds=selected_deepstack,
        )
        trimmed = [generated_ids[0]]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def _answer_or_route_from_recent_context(
        self,
        selection: Dict[str, Any],
        prompt: str,
        *,
        source: str,
    ) -> Optional[str]:
        if self._task_gate_mode() != "recent_context_sufficiency":
            return None

        tokenizer = self.processor.tokenizer
        retrieve_id, sufficient_id = resolve_query_choice_token_ids(tokenizer)
        raw_question = str(
            getattr(self.owner, "_task_gate_query_text", "") or ""
        ).strip()
        gate_question = raw_question or prompt
        router_prompt = build_recent_context_sufficiency_prompt(gate_question)
        threshold = float(self.config.get("task_gate_recent_sufficiency_threshold", 0.0))
        routed: Dict[str, Any] = {}

        def route_prefill(
            generate_kwargs: Dict[str, Any],
            *,
            prompt_len: int,
            **_unused: Any,
        ) -> torch.Tensor:
            forward_kwargs = {
                key: value
                for key, value in generate_kwargs.items()
                if key in {"inputs_embeds", "attention_mask", "position_ids"}
            }
            relative_positions = self._rope_position_mode() == "relative"
            if relative_positions:
                # Match the normal selected-prompt generate path. Qwen's
                # prepare_inputs_for_generation discards explicit MRoPE ids in
                # relative mode and prefills with scalar cache positions.
                forward_kwargs.pop("position_ids", None)
                forward_kwargs["cache_position"] = torch.arange(
                    prompt_len,
                    dtype=torch.long,
                    device=generate_kwargs["inputs_embeds"].device,
                )
            outputs = self.model(
                **forward_kwargs,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            next_logits = outputs.logits[0, -1].float()
            retrieve_logit = float(next_logits[retrieve_id].item())
            sufficient_logit = float(next_logits[sufficient_id].item())
            score = retrieve_logit - sufficient_logit
            retrieve = score >= threshold
            cache = outputs.past_key_values
            cache_length = int(cache.get_seq_length())
            decision = {
                "enabled": True,
                "mode": "recent_context_sufficiency",
                "source": source,
                "predicted_task_type": "backward" if retrieve else "realtime",
                "selected_policy": "retrieval" if retrieve else "recent_only",
                "retrieval_enabled": retrieve,
                "score": score,
                "threshold": threshold,
                "retrieve_logit": retrieve_logit,
                "sufficient_logit": sufficient_logit,
                "choice_logits": {
                    "retrieval": retrieve_logit,
                    "recent_only": sufficient_logit,
                },
                "gate_model_forward_count": 1,
                "recent_units": len(selection.get("recent", [])),
                "recent_unit_ids": [
                    int(frame.frame_id) for frame in selection.get("recent", [])
                ],
                "recent_source_frame_count": len(selection.get("recent_source_frames", [])),
                "prefill_tokens": int(prompt_len),
                "cache_tokens": cache_length,
                "cache_reused_for_answer": not retrieve,
                "task_gate_input_source": (
                    "question_text" if raw_question else "full_prompt"
                ),
            }
            self._set_gate_decision(decision)
            routed["retrieve"] = retrieve
            if retrieve:
                return torch.tensor([[sufficient_id]], dtype=torch.long, device=next_logits.device)

            if not hasattr(cache, "crop"):
                raise RuntimeError("recent-context sufficiency requires a crop-capable KV cache")
            answer_prompt = prompt
            router_ids = tokenizer.encode(router_prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_prompt, add_special_tokens=False)
            common_tokens = 0
            for router_id, answer_id in zip(router_ids, answer_ids):
                if router_id != answer_id:
                    break
                common_tokens += 1
            answer_prefix = getattr(self.owner, "assistant_answer_prefix", None)
            answer_prefix_text = str(answer_prefix() or "") if callable(answer_prefix) else ""
            answer_turn_ids = [
                int(self.owner.im_end_id),
                *tokenizer.encode("\n", add_special_tokens=False),
                int(self.owner.im_start_id),
                *tokenizer.encode("assistant\n", add_special_tokens=False),
                *tokenizer.encode(answer_prefix_text, add_special_tokens=False),
            ]
            crop_tokens = len(router_ids) - common_tokens + len(answer_turn_ids)
            answer_cache_length = cache_length - crop_tokens
            if common_tokens <= 0 or answer_cache_length <= 0:
                raise RuntimeError("recent-context gate prompt has no reusable answer prefix")
            cache.crop(answer_cache_length)
            continuation_ids = [
                *answer_ids[common_tokens:],
                *answer_turn_ids,
            ]
            if relative_positions:
                continuation_position_ids = torch.arange(
                    answer_cache_length,
                    answer_cache_length + len(continuation_ids),
                    dtype=torch.long,
                    device=next_logits.device,
                ).view(1, 1, -1).expand(3, 1, -1)
            else:
                prefill_position_ids = forward_kwargs.get("position_ids")
                if not isinstance(prefill_position_ids, torch.Tensor):
                    raise RuntimeError("recent-context gate prefill is missing position_ids")
                if int(prefill_position_ids.shape[-1]) != cache_length:
                    raise RuntimeError("recent-context gate cache and position lengths differ")
                continuation_offsets = torch.arange(
                    1,
                    len(continuation_ids) + 1,
                    dtype=torch.long,
                    device=prefill_position_ids.device,
                ).view(1, 1, -1)
                continuation_position_ids = (
                    prefill_position_ids[..., answer_cache_length - 1 : answer_cache_length]
                    + continuation_offsets
                )
            continuation = torch.tensor(
                [continuation_ids], dtype=torch.long, device=next_logits.device
            )
            attention_mask = torch.ones(
                (1, answer_cache_length + len(continuation_ids)),
                dtype=torch.long,
                device=next_logits.device,
            )
            answer_kwargs: Dict[str, Any] = {
                "input_ids": continuation,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "cache_position": torch.arange(
                    answer_cache_length,
                    answer_cache_length + len(continuation_ids),
                    dtype=torch.long,
                    device=next_logits.device,
                ),
                "max_new_tokens": _as_int(self.config, "max_new_tokens"),
                "do_sample": _as_bool(self.config, "do_sample"),
                "logits_to_keep": 1,
            }
            if answer_kwargs["do_sample"]:
                answer_kwargs["temperature"] = max(
                    float(self.config.get("decode_temperature", 1.0)), 1e-6
                )
            generated = self._generate_selected_prompt_ids(
                answer_kwargs,
                prompt_len=len(continuation_ids),
                outputs_include_prompt=True,
                language_position_ids=continuation_position_ids,
                position_prompt_len=answer_cache_length + len(continuation_ids),
            )
            return generated[:, len(continuation_ids):]

        answer = self._generate_from_selected_prompt(
            selection,
            router_prompt,
            prefill_callback=route_prefill,
        )
        if not routed:
            raise RuntimeError("recent-context sufficiency gate did not run")
        return None if routed["retrieve"] else answer

    def _answer_stream_question(
        self,
        session: _Qwen3VLStreamSession,
        prompt: str,
    ) -> str:
        self._last_query_phase_timing = {}
        layers, rotary_emb, norm, lm_head, embed_tokens = self._language_parts()
        prune_layer = self._resolve_prune_layer(layers)
        lower_layer_ids = list(range(prune_layer))
        query_layer = prune_layer - 1
        prefix_ids, suffix_ids = self._video_prompt_wrapper_token_ids(prompt, session.video_key)
        prefix_len = len(prefix_ids)
        suffix_len = len(suffix_ids)
        embed_device = self._layer_device(embed_tokens, self.owner._input_device())
        fullkv_mode = _full_kv_enabled(self.config)
        preserve_fullkv_stream = bool(
            fullkv_mode
            and _as_bool(self.config, "fullkv_preserve_stream_history")
        )

        if self._task_gate_mode() == "recent_context_sufficiency":
            gate_selection = self._recent_context_gate_selection(
                session.frame_states
            )
            gate_selection = self._attach_recent_source_frames(
                session, gate_selection
            )
            recent_answer = self._answer_or_route_from_recent_context(
                gate_selection,
                prompt,
                source="stream_recent_prefill",
            )
            if recent_answer is not None:
                return recent_answer

        if preserve_fullkv_stream:
            archived_lower_kv = {}
            for layer_idx, entry in session.raw_lower_kv.items():
                archived_lower_kv[int(layer_idx)] = {
                    "k": entry["k"].detach().to(device="cpu"),
                    "v": entry["v"].detach().to(device="cpu"),
                    "positions": entry["positions"].detach().to(device="cpu"),
                }
            if len(archived_lower_kv) != prune_layer:
                raise RuntimeError(
                    "Persistent FullKV Qwen3-VL archive does not cover every decoder layer: "
                    f"archive={len(archived_lower_kv)}, expected={prune_layer}"
                )
            session.raw_lower_kv = archived_lower_kv
            session.active_lower_kv = archived_lower_kv

        question_hidden_for_query = None
        question_query_cache = None
        prefix_hidden = None
        prefix_lower_cache = None
        prefix_positions = None
        if fullkv_mode:
            if session.clusters:
                raise RuntimeError("FullKV Qwen3-VL must not contain compressed clusters")
            all_frames = list(session.frame_states)
            selection = {
                "query_vec": None,
                "short": all_frames,
                "clusters": [],
                "all_clusters": [],
                "retrieved": [],
                "retrieved_seed": [],
                "recent": all_frames,
                "search_candidates": [],
                "short_scores": [],
                "search_start": 0,
                "search_end": 0,
                "token_selection_stats": {
                    "retrieval_selection_granularity": "none",
                    "memory_policy": "full_kv",
                },
                "probe_query_vecs": {},
            }
        else:
            if suffix_len <= 0:
                raise RuntimeError("Qwen3-VL streaming prompt wrapper has an empty question suffix")
            sync_latency = _as_bool(self.config, "latency_sync_cuda")
            _sync_cuda_if_requested(sync_latency)
            self._last_query_phase_timing["query_encoding_started_time"] = time.perf_counter()
            # Prefill the exact suffix used by the final multimodal prompt.
            # Its shallow KV/hidden state is retained for post-retrieval reuse.
            question_embeds = self._embed_token_ids(suffix_ids, embed_tokens, embed_device)
            text_inputs = {
                "input_ids": torch.tensor(
                    [suffix_ids],
                    dtype=torch.long,
                    device=question_embeds.device,
                )
            }
            question_positions = self._make_scalar_positions(
                session.next_position,
                int(question_embeds.shape[1]),
                question_embeds.device,
            )
            question_context_kwargs = self._question_query_context_kwargs(
                question_len=int(question_embeds.shape[1]),
                visual_window_tokens=int(session.local_window_tokens),
                sink_len=int(session.sink_len),
                sink_raw_kv=session.sink_lower_kv,
            )
            capture_q_layers = {query_layer}
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
                question_hidden_for_query, question_query_cache = self._forward_lower_layers_raw(
                    hidden_states=question_embeds,
                    layers=layers,
                    rotary_emb=rotary_emb,
                    start_layer=0,
                    end_layer=prune_layer,
                    past_raw_kv=session.raw_lower_kv,
                    positions=question_positions,
                    update_cache=False,
                    capture_q_layers=capture_q_layers,
                    cache_policy="capture_current",
                    **question_context_kwargs,
                )
            query_vec = self._query_vector_from_current_q(
                question_query_cache,
                query_layer,
            )
            probe_query_vecs = self._task_gate_probe_vectors(
                prompt=prompt,
                layers=layers,
                rotary_emb=rotary_emb,
                query_layer=query_layer,
                context_raw_kv=session.raw_lower_kv,
                position_start=int(session.next_position),
                context_sink_len=int(session.sink_len),
                context_sink_raw_kv=session.sink_lower_kv,
                context_local_window_tokens=int(session.local_window_tokens),
            )
            attention_features = None
            attention_observation_features = None
            token_vote_query_indices = None
            if token_vote_enabled:
                token_vote_query_indices = self._retrieval_vote_query_indices(
                    text_inputs,
                    prompt,
                )
            if attention_layer is not None:
                query_indices = self._attention_distribution_query_indices(
                    text_inputs,
                    prompt,
                )
                attention_observation_features = {}
                for layer_idx in attention_observation_layers:
                    _layer, layer_features = self._build_attention_distribution_features(
                        q_entry=question_query_cache[layer_idx],
                        k_entry=session.raw_lower_kv[layer_idx],
                        query_indices=query_indices,
                        frames=session.frame_states,
                        rotary_emb=rotary_emb,
                        layers=layers,
                        layer_idx=layer_idx,
                    )
                    attention_observation_features[layer_idx] = layer_features
                attention_features = attention_observation_features[attention_layer]
            _sync_cuda_if_requested(sync_latency)
            self._last_query_phase_timing["query_encoding_finished_time"] = time.perf_counter()
            selection = self._select_stream_memory(
                session,
                query_vec,
                query_layer,
                prompt=prompt,
                probe_query_vecs=probe_query_vecs,
                attention_distribution_layer=attention_layer,
                attention_distribution_features=attention_features,
                attention_distribution_observation_features=attention_observation_features,
                token_vote_query_cache=question_query_cache,
                token_vote_query_indices=token_vote_query_indices,
                rotary_emb=rotary_emb,
                layers=layers,
            )
        if self._selected_generate_mode() == "simple_prompt":
            if _as_bool(self.config, "debug"):
                simple_stats = {
                    "selected_generate_mode": "simple_prompt",
                    "stream_mode": "open_window",
                    "rekv_sink_path_stats": dict(getattr(self, "_rekv_sink_path_stats", {})),
                    "selected_short_units": [frame.frame_id for frame in selection.get("short", [])],
                    "selected_short_visual_tokens": sum(
                        int(frame.token_indices.numel()) for frame in selection.get("short", [])
                    ),
                    "recent_units": [frame.frame_id for frame in selection.get("recent", [])],
                    "retrieved_short_units": [frame.frame_id for frame in selection.get("retrieved", [])],
                }
                simple_stats.update(selection.get("token_selection_stats") or {})
                print(f"[{MODEL_NAME}] simple_prompt_selection_stats={json.dumps(simple_stats, ensure_ascii=False)}", flush=True)
            return self._generate_from_selected_prompt(selection, prompt)
        if not fullkv_mode:
            if prefix_len <= 0 or prefix_len != int(session.prompt_prefix_len):
                raise RuntimeError(
                    "Qwen3-VL final prompt prefix does not match the shallow-prefilled stream prefix"
                )
            if not isinstance(session.prompt_prefix_hidden_after_prune, torch.Tensor):
                raise RuntimeError("Qwen3-VL stream is missing its prefetched prefix hidden state")
            if not session.prompt_prefix_lower_kv:
                raise RuntimeError("Qwen3-VL stream is missing its prefetched prefix KV")
            prefix_positions = self._make_scalar_positions(0, prefix_len, embed_device)
            prefix_hidden = session.prompt_prefix_hidden_after_prune
            prefix_lower_cache = {}
            for layer_idx in lower_layer_ids:
                entry = session.prompt_prefix_lower_kv.get(layer_idx)
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"Qwen3-VL stream is missing prefetched prefix KV at layer {layer_idx}"
                    )
                prefix_lower_cache[layer_idx] = {
                    "k": entry["k"],
                    "v": entry["v"],
                    "positions": prefix_positions.to(device=entry["k"].device, dtype=torch.long),
                }
        (
            selected_visual_cache,
            selected_visual_hidden,
            selected_visual_positions,
            selected_visual_mask,
            selected_deepstack,
            stats,
        ) = self._build_stream_selected_visual_cache(session, selection, lower_layer_ids)

        visual_len = (
            int(stats["stream_selected_visual_tokens"])
            if fullkv_mode
            else int(selected_visual_hidden.shape[1])
        )
        fullkv_visual_last = None
        if fullkv_mode:
            if visual_len <= 0:
                raise RuntimeError("FullKV Qwen3-VL requires a non-empty visual context")
            # All decoder layers have already processed the visual stream. Only
            # the final prompt position is consumed below, so keep one exact
            # hidden vector as a fallback and release otherwise-dead full-video
            # hidden/deepstack state before rebuilding the prompt cache.
            fullkv_visual_last = selected_visual_hidden[:, -1:, :].clone()
            selected_visual_hidden = fullkv_visual_last
            selected_visual_mask = None
            selected_deepstack = None
            if not preserve_fullkv_stream:
                session.hidden_after_prune = None
                session.visual_pos_masks = None
                session.deepstack_visual_embeds = None
                session.frame_states.clear()
                session.raw_lower_kv.clear()
                session.active_lower_kv.clear()
        if fullkv_mode:
            if prefix_len > 0:
                prefix_embeds = self._embed_token_ids(prefix_ids, embed_tokens, embed_device)
                prefix_positions = self._make_scalar_positions(0, prefix_len, embed_device)
                with torch.no_grad():
                    prefix_hidden, prefix_lower_cache = self._forward_lower_layers_raw(
                        hidden_states=prefix_embeds,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        start_layer=0,
                        end_layer=prune_layer,
                        past_raw_kv={},
                        positions=prefix_positions,
                        update_cache=True,
                        sink_len=prefix_len,
                        local_window_tokens=prefix_len,
                    )
            else:
                prefix_hidden = selected_visual_hidden[:, :0, :]
                prefix_positions = self._make_scalar_positions(
                    0,
                    0,
                    selected_visual_hidden.device,
                )
                prefix_lower_cache = {}

            selected_visual_cache, selected_visual_positions = self._offset_stream_visual_context(
                visual_cache=selected_visual_cache,
                visual_positions=selected_visual_positions,
                prefix_len=prefix_len,
                lower_layer_ids=lower_layer_ids,
            )
            if prefix_len > 0:
                prefix_visual_cache = self._concat_lower_caches(
                    prefix_lower_cache,
                    selected_visual_cache,
                    lower_layer_ids,
                    consume_right_cache=True,
                )
            else:
                prefix_visual_cache = selected_visual_cache

            suffix_start = prefix_len + visual_len
            suffix_positions = self._make_scalar_positions(
                suffix_start,
                suffix_len,
                embed_device,
            )
            if suffix_len > 0:
                suffix_embeds = self._embed_token_ids(suffix_ids, embed_tokens, embed_device)
                with torch.no_grad():
                    suffix_hidden, selected_lower_cache = self._forward_lower_layers_raw(
                        hidden_states=suffix_embeds,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        start_layer=0,
                        end_layer=prune_layer,
                        past_raw_kv=prefix_visual_cache,
                        positions=suffix_positions,
                        update_cache=True,
                        consume_past_cache=True,
                        sink_len=prefix_len,
                        local_window_tokens=prefix_len + visual_len + suffix_len,
                    )
            else:
                suffix_hidden = selected_visual_hidden[:, :0, :]
                selected_lower_cache = prefix_visual_cache

            if suffix_len > 0:
                selected_hidden = suffix_hidden[:, -1:, :]
                selected_positions = suffix_positions[..., -1:]
            elif prefix_len > 0:
                selected_hidden = prefix_hidden[:, -1:, :]
                selected_positions = prefix_positions[..., -1:]
            else:
                assert fullkv_visual_last is not None
                selected_hidden = fullkv_visual_last
                selected_positions = self._make_scalar_positions(
                    visual_len - 1,
                    1,
                    selected_hidden.device,
                )
            selected_masks = torch.zeros(
                (1, 1),
                device=selected_hidden.device,
                dtype=torch.bool,
            )
            next_position = prefix_len + visual_len + suffix_len
        else:
            assert prefix_hidden is not None
            assert prefix_lower_cache is not None
            assert prefix_positions is not None
            assert question_hidden_for_query is not None
            assert question_query_cache is not None
            (
                selected_lower_cache,
                selected_hidden,
                selected_positions,
                selected_masks,
            ) = self._compose_reused_stream_shallow_context(
                prefix_hidden=prefix_hidden,
                prefix_lower_cache=prefix_lower_cache,
                prefix_positions=prefix_positions,
                selected_visual_cache=selected_visual_cache,
                selected_visual_hidden=selected_visual_hidden,
                selected_visual_positions=selected_visual_positions,
                selected_visual_mask=selected_visual_mask,
                question_hidden=question_hidden_for_query,
                question_query_cache=question_query_cache,
                lower_layer_ids=lower_layer_ids,
            )
            next_position = (
                int(selected_positions[0].max().item()) + 1
                if selected_positions.numel() > 0
                else 0
            )
        stats.update(
            {
                "stream_prompt_prefix_tokens": prefix_len,
                "stream_prompt_suffix_tokens": suffix_len,
                "stream_selected_context_tokens": (
                    prefix_len + visual_len + suffix_len
                    if fullkv_mode
                    else int(selected_hidden.shape[1])
                ),
                "stream_selected_position_min": int(selected_positions[0].min().item()) if selected_positions.numel() else 0,
                "stream_selected_position_max": int(selected_positions[0].max().item()) if selected_positions.numel() else 0,
                "stream_next_position": next_position,
            }
        )

        if _as_bool(self.config, "debug"):
            print(f"[{MODEL_NAME}] stream_internal_kv_stats={json.dumps(stats, ensure_ascii=False)}", flush=True)

        return self._decode_from_selected_cache(
            selected_lower_cache=selected_lower_cache,
            selected_hidden=selected_hidden,
            selected_positions=selected_positions,
            next_position=next_position,
            selected_visual_pos_masks=selected_masks,
            selected_deepstack_visual_embeds=selected_deepstack,
            layers=layers,
            rotary_emb=rotary_emb,
            norm=norm,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            prune_layer=prune_layer,
            sink_len=prefix_len,
            local_window_tokens=(
                next_position + _as_int(self.config, "max_new_tokens")
                if _full_kv_enabled(self.config)
                else int(session.local_window_tokens)
            ),
        )
