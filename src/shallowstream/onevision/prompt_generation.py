from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image


class OneVisionPromptGenerationMixin:
    """Rebuild a compact selected prompt and prefill it through every LM layer."""

    def _visual_encoder_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        get_vision_tower = getattr(self.model, "get_vision_tower", None)
        vision_tower = get_vision_tower() if callable(get_vision_tower) else None
        if vision_tower is None:
            vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is None:
            raise RuntimeError("OneVision model does not expose its vision tower")
        for parameter in vision_tower.parameters():
            return parameter.device, parameter.dtype
        for buffer in vision_tower.buffers():
            return buffer.device, buffer.dtype
        raise RuntimeError("OneVision vision tower has no parameters or buffers")

    @torch.inference_mode()
    def _encode_images_for_selected_prompt(
        self,
        images: Sequence[Image.Image],
    ) -> List[torch.Tensor]:
        """Run retained RGB frames through OneVision's native AnyRes image path."""
        if not images:
            return []
        if not hasattr(self.model, "get_image_features"):
            raise RuntimeError(
                "OneVision model does not expose get_image_features for image re-encoding"
            )

        vision_device, vision_dtype = self._visual_encoder_device_dtype()
        encoded_images: List[torch.Tensor] = []
        for image in images:
            image_inputs = self.processor.image_processor(
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
            pixel_values = image_inputs.get("pixel_values")
            image_sizes = image_inputs.get("image_sizes")
            if not isinstance(pixel_values, torch.Tensor) or not isinstance(
                image_sizes,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "OneVision image_processor must return pixel_values and image_sizes tensors"
                )
            image_features = self.model.get_image_features(
                pixel_values=pixel_values.to(
                    device=vision_device,
                    dtype=vision_dtype,
                ),
                image_sizes=image_sizes.to(device=vision_device),
            )
            if not isinstance(image_features, (list, tuple)) or len(image_features) != 1:
                raise RuntimeError(
                    "OneVision get_image_features must return one feature tensor per image"
                )
            image_embeds = image_features[0]
            if not isinstance(image_embeds, torch.Tensor) or image_embeds.ndim != 2:
                shape = (
                    tuple(image_embeds.shape)
                    if isinstance(image_embeds, torch.Tensor)
                    else ""
                )
                raise RuntimeError(
                    "OneVision image features must have shape (tokens, hidden), "
                    f"got {type(image_embeds)} {shape}"
                )
            encoded_images.append(image_embeds)
        return encoded_images

    def _selected_generate_mode(self) -> str:
        mode = str(self.config.get("selected_generate_mode", "simple_prompt")).strip().lower()
        if mode not in {"simple_prompt", "internal_kv"}:
            raise ValueError(
                "selected_generate_mode must be 'simple_prompt' or 'internal_kv', "
                f"got {mode!r}"
            )
        return mode

    def _build_selected_prompt_embeds(
        self,
        *,
        prompt_ids: torch.Tensor,
        selected_frames: List[int],
        selected_long_cluster_ids: Optional[List[int]],
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        layers, embed_tokens, _norm, _lm_head = self._get_lm_components()
        init_input_embeds = self.state.get("init_input_embeds")
        if not isinstance(init_input_embeds, torch.Tensor) or init_input_embeds.numel() == 0:
            raise RuntimeError("OneVision simple_prompt requires retained init input embeddings")

        dtype = init_input_embeds.dtype
        parts: List[torch.Tensor] = [
            init_input_embeds.reshape(-1, init_input_embeds.shape[-1]).to(
                device=device,
                dtype=dtype,
            )
        ]
        use_labels = bool(self.config.get("selected_prompt_use_labels", False))
        selected_label_tokens = 0

        def append_label(text: str) -> None:
            nonlocal selected_label_tokens
            if not use_labels:
                return
            label_ids = self.tokenizer(
                f"{text}\n",
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
            if label_ids.numel() <= 0:
                return
            label_embeds = embed_tokens(label_ids).to(device=device, dtype=dtype)
            parts.append(label_embeds[0])
            selected_label_tokens += int(label_embeds.shape[1])

        long_clusters = self.state.get("long_clusters")
        if not isinstance(long_clusters, list):
            long_clusters = []
        valid_long_cluster_ids = sorted(
            {
                int(cluster_id)
                for cluster_id in (selected_long_cluster_ids or [])
                if 0 <= int(cluster_id) < len(long_clusters)
            }
        )
        frame_input_embeds = self.state.get("frame_input_embeds")
        if not isinstance(frame_input_embeds, torch.Tensor):
            raise RuntimeError("OneVision simple_prompt requires retained frame input embeddings")
        total_frames = int(frame_input_embeds.shape[0])
        valid_frames = sorted(
            {int(frame_id) for frame_id in selected_frames if 0 <= int(frame_id) < total_frames}
        )
        if len(valid_frames) != len(set(int(frame_id) for frame_id in selected_frames)):
            raise ValueError(
                "OneVision simple_prompt received frame ids outside retained short memory: "
                f"selected={selected_frames}, retained={total_frames}"
            )

        frame_source_ids = self.state.get("frame_source_ids")
        frame_captions = self.state.get("frame_captions")
        last_selection = self.state.get("last_selection")
        recent_ids = {
            int(frame_id)
            for frame_id in (
                last_selection.get("recent_idx", [])
                if isinstance(last_selection, dict)
                else []
            )
        }
        retrieved_frames = [
            frame_id for frame_id in valid_frames if frame_id not in recent_ids
        ]
        recent_frames = [
            frame_id for frame_id in valid_frames if frame_id in recent_ids
        ]

        reencode_all_as_images = bool(
            self.config.get("selected_prompt_reencode_all_as_images", False)
        )
        reencode_recent_as_images = bool(
            self.config.get("selected_prompt_reencode_recent_as_images", False)
        )
        if reencode_all_as_images and reencode_recent_as_images:
            raise ValueError(
                "OneVision simple_prompt image modes are mutually exclusive: "
                "enable either selected_prompt_reencode_all_as_images or "
                "selected_prompt_reencode_recent_as_images"
            )
        long_image_embeds: Dict[int, torch.Tensor] = {}
        frame_image_embeds: Dict[int, torch.Tensor] = {}
        if reencode_all_as_images or reencode_recent_as_images:
            frame_source_images = self.state.get("frame_source_images")
            if not isinstance(frame_source_images, list) or len(frame_source_images) != total_frames:
                raise RuntimeError(
                    "OneVision image-reencode simple_prompt requires source images aligned with "
                    f"short memory: images={len(frame_source_images) if isinstance(frame_source_images, list) else 'missing'}, "
                    f"frames={total_frames}"
                )

            ordered_images: List[Image.Image] = []
            ordered_keys: List[Tuple[str, int]] = []
            if reencode_all_as_images:
                for cluster_id in valid_long_cluster_ids:
                    source_image = long_clusters[cluster_id].get("source_image")
                    if not isinstance(source_image, Image.Image):
                        raise RuntimeError(
                            "OneVision all-image simple_prompt selected a long cluster without "
                            f"a representative source image: cluster_id={cluster_id}"
                        )
                    ordered_images.append(source_image)
                    ordered_keys.append(("long", cluster_id))

            image_frame_ids = (
                [*retrieved_frames, *recent_frames]
                if reencode_all_as_images
                else recent_frames
            )
            for frame_id in image_frame_ids:
                source_image = frame_source_images[frame_id]
                if not isinstance(source_image, Image.Image):
                    raise RuntimeError(
                        "OneVision image-reencode simple_prompt source buffer contains a non-PIL "
                        f"entry: frame_id={frame_id}"
                    )
                ordered_images.append(source_image)
                ordered_keys.append(("short", frame_id))

            encoded_images = self._encode_images_for_selected_prompt(ordered_images)
            if len(encoded_images) != len(ordered_keys):
                raise RuntimeError(
                    "OneVision image encoder returned an unexpected number of feature groups: "
                    f"encoded={len(encoded_images)}, requested={len(ordered_keys)}"
                )
            for (memory_kind, memory_id), image_embeds in zip(
                ordered_keys,
                encoded_images,
            ):
                if memory_kind == "long":
                    long_image_embeds[memory_id] = image_embeds
                else:
                    frame_image_embeds[memory_id] = image_embeds

        selected_long_visual_tokens = 0
        selected_long_image_visual_tokens = 0
        if valid_long_cluster_ids:
            append_label("Compressed historical video memory:")
        for cluster_id in valid_long_cluster_ids:
            cluster_embeds = long_image_embeds.get(cluster_id)
            if cluster_embeds is None:
                cluster_embeds = long_clusters[cluster_id].get("input_embeds")
            if not isinstance(cluster_embeds, torch.Tensor) or cluster_embeds.numel() == 0:
                raise RuntimeError(
                    "OneVision simple_prompt selected a long cluster without retained visual inputs: "
                    f"cluster_id={cluster_id}"
                )
            cluster_embeds = cluster_embeds.reshape(-1, cluster_embeds.shape[-1])
            parts.append(cluster_embeds.to(device=device, dtype=dtype))
            token_count = int(cluster_embeds.shape[0])
            selected_long_visual_tokens += token_count
            if cluster_id in long_image_embeds:
                selected_long_image_visual_tokens += token_count

        selected_caption_tokens = 0
        selected_short_visual_tokens = 0
        selected_retrieved_image_visual_tokens = 0
        selected_recent_image_visual_tokens = 0

        def append_frames(frame_ids: List[int], *, recent: bool, label: str) -> None:
            nonlocal selected_caption_tokens
            nonlocal selected_short_visual_tokens
            nonlocal selected_retrieved_image_visual_tokens
            nonlocal selected_recent_image_visual_tokens
            if not frame_ids:
                return
            append_label(label)
            for frame_id in frame_ids:
                visual_embeds = frame_image_embeds.get(frame_id)
                if visual_embeds is None:
                    visual_embeds = frame_input_embeds[frame_id]
                visual_embeds = visual_embeds.reshape(-1, visual_embeds.shape[-1])
                parts.append(visual_embeds.to(device=device, dtype=dtype))
                token_count = int(visual_embeds.shape[0])
                selected_short_visual_tokens += token_count
                if frame_id in frame_image_embeds:
                    if recent:
                        selected_recent_image_visual_tokens += token_count
                    else:
                        selected_retrieved_image_visual_tokens += token_count

                frame_ts = (
                    float(frame_source_ids[frame_id])
                    if isinstance(frame_source_ids, list)
                    and frame_id < len(frame_source_ids)
                    else 0.0
                )
                caption = (
                    str(frame_captions[frame_id])
                    if isinstance(frame_captions, list)
                    and frame_id < len(frame_captions)
                    else ""
                )
                caption_ids = self._caption_token_ids_for_frame(
                    frame_ts,
                    caption,
                    device,
                )
                if isinstance(caption_ids, torch.Tensor) and caption_ids.numel() > 0:
                    caption_embeds = embed_tokens(caption_ids).to(
                        device=device,
                        dtype=dtype,
                    )
                    parts.append(caption_embeds[0])
                    selected_caption_tokens += int(caption_embeds.shape[1])

        append_frames(
            retrieved_frames,
            recent=False,
            label="Retrieved short video memory:",
        )
        append_frames(
            recent_frames,
            recent=True,
            label="Recent video memory:",
        )

        question_embeds = embed_tokens(prompt_ids).to(device=device, dtype=dtype)
        parts.append(question_embeds[0])
        prompt_embeds = torch.cat(parts, dim=0).unsqueeze(0)
        stats = {
            "selected_generate_mode": "simple_prompt",
            "selected_short_frames": valid_frames,
            "selected_retrieved_short_frames": retrieved_frames,
            "selected_recent_frames": recent_frames,
            "selected_long_clusters": valid_long_cluster_ids,
            "selected_short_visual_tokens": int(selected_short_visual_tokens),
            "selected_long_visual_tokens": int(selected_long_visual_tokens),
            "selected_long_image_visual_tokens": int(selected_long_image_visual_tokens),
            "selected_retrieved_image_visual_tokens": int(
                selected_retrieved_image_visual_tokens
            ),
            "selected_recent_image_visual_tokens": int(
                selected_recent_image_visual_tokens
            ),
            "selected_prompt_reencode_all_as_images": reencode_all_as_images,
            "selected_prompt_reencode_recent_as_images": reencode_recent_as_images,
            "selected_caption_tokens": int(selected_caption_tokens),
            "question_tokens": int(prompt_ids.shape[1]),
            "reprefill_tokens": int(prompt_embeds.shape[1]),
            "reprefill_layer_count": int(len(layers)),
        }
        self.state["last_simple_prompt_stats"] = stats
        return prompt_embeds, stats

    def _decode_from_selected_prompt(
        self,
        *,
        prompt_ids: torch.Tensor,
        selected_frames: List[int],
        selected_long_cluster_ids: Optional[List[int]],
        device: str,
    ) -> Tuple[str, float, float, int]:
        layers, embed_tokens, norm, lm_head = self._get_lm_components()
        if lm_head is None:
            raise RuntimeError("lm_head is None")

        with torch.inference_mode():
            t0 = time.perf_counter()
            prompt_embeds, _stats = self._build_selected_prompt_embeds(
                prompt_ids=prompt_ids,
                selected_frames=selected_frames,
                selected_long_cluster_ids=selected_long_cluster_ids,
                device=device,
            )
            if prompt_embeds.device.type == "cuda":
                free_bytes, _total_bytes = torch.cuda.mem_get_info(prompt_embeds.device)
                if free_bytes < 4 * 1024**3:
                    # Lower prefill temporaries are dead at this phase boundary;
                    # release their inactive blocks before the answer MLP runs.
                    torch.cuda.empty_cache()
            hidden, full_cache = self._forward_layer_range(
                prompt_embeds,
                0,
                len(layers),
                {},
                causal=True,
            )
            if norm is not None:
                hidden = norm(hidden)
            # Generation only needs the next-token distribution.  Projecting
            # every multimodal prompt position to the full vocabulary creates
            # a multi-gigabyte logits tensor for AnyRes recent images without
            # changing the generated token.
            logits = lm_head(hidden[:, -1:, :])[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if bool(self.config.get("latency_sync_cuda")) and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(device))
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            generated = [next_token]
            eos_id = self.tokenizer.eos_token_id
            eos_set = set(eos_id) if isinstance(eos_id, list) else {eos_id}
            for _ in range(max(int(self.config["max_new_tokens"]) - 1, 0)):
                if (
                    not bool(self.config.get("force_exact_new_tokens"))
                    and int(next_token.item()) in eos_set
                ):
                    break
                token_embed = embed_tokens(next_token).to(
                    device=device,
                    dtype=prompt_embeds.dtype,
                )
                hidden, full_cache = self._forward_layer_range(
                    token_embed,
                    0,
                    len(layers),
                    full_cache,
                    causal=True,
                )
                if norm is not None:
                    hidden = norm(hidden)
                logits = lm_head(hidden[:, -1:, :])[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated.append(next_token)

            out_ids = torch.cat([prompt_ids, *generated], dim=1)
            text = self._decode_generated(out_ids, prompt_ids.shape[1])
            decode_total_ms = (time.perf_counter() - t0) * 1000.0
            return text, ttft_ms, decode_total_ms, len(generated)
