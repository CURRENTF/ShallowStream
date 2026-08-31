"""Qwen3VLStreamingMixin implementation."""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from src.shallowstream.common import select_temporal_retrieval_ids

from .config import (
    MODEL_NAME,
    _as_bool,
    _as_float,
    _as_int,
    _full_kv_enabled,
    _retrieval_expansion_strategy,
)
from .state import (
    _FrameKVState,
    _LongKVCluster,
    _Qwen3VLStreamSession,
    _RecentSourceFrame,
)


class Qwen3VLStreamingMixin:

    def _recent_source_frame_capacity(self) -> int:
        temporal_factor = max(_as_int(self.config, "video_temporal_patch_size"), 1)
        retrieval_units = self._retrieval_recent_units()
        realtime_units_raw = self.config.get("task_gate_realtime_recent_units")
        realtime_units = (
            retrieval_units
            if realtime_units_raw is None
            else max(int(realtime_units_raw), 0)
        )
        return max(retrieval_units, realtime_units) * temporal_factor

    def _append_recent_source_frames(
        self,
        session: _Qwen3VLStreamSession,
        frames: Sequence[Any],
    ) -> None:
        capacity = self._recent_source_frame_capacity()
        if capacity <= 0:
            session.recent_source_frames = []
            return

        for frame in frames:
            image = getattr(frame, "image", None)
            if not isinstance(image, Image.Image):
                continue
            timestamp = float(getattr(frame, "timestamp"))
            sample_index = int(getattr(frame, "index"))
            retained = _RecentSourceFrame(
                sample_index=sample_index,
                timestamp=timestamp,
                image=image,
            )
            session.recent_source_frames.append(retained)

        if len(session.recent_source_frames) > capacity:
            session.recent_source_frames = session.recent_source_frames[-capacity:]

    def _attach_recent_source_frames(
        self,
        session: _Qwen3VLStreamSession,
        selection: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected = dict(selection)
        temporal_factor = max(_as_int(self.config, "video_temporal_patch_size"), 1)
        source_count = len(selected.get("recent", [])) * temporal_factor
        recent_source_frames = (
            list(session.recent_source_frames[-source_count:])
            if source_count > 0
            else []
        )
        selected["recent_source_frames"] = recent_source_frames
        selected["token_selection_stats"] = dict(
            selected.get("token_selection_stats") or {}
        )
        selected["token_selection_stats"].update(
            {
                "recent_source_frame_count": len(recent_source_frames),
                "recent_source_frame_sample_indices": [
                    int(frame.sample_index) for frame in recent_source_frames
                ],
                "recent_source_frame_timestamps": [
                    float(frame.timestamp) for frame in recent_source_frames
                ],
            }
        )
        return selected

    def _get_text_model_for_generate(self):
        return getattr(self.owner, "_text_model", getattr(self.model, "model", self.model))

    def _unit_grid_rows(self, inputs: Dict[str, Any], unit_count: int) -> List[torch.Tensor]:
        unit_count = max(int(unit_count), 0)
        grid = inputs.get("video_grid_thw")
        if isinstance(grid, torch.Tensor) and grid.numel() >= 3:
            row = grid.detach().cpu().reshape(-1, 3)[0].long()
            h = max(int(row[1].item()), 1)
            w = max(int(row[2].item()), 1)
            return [torch.tensor([1, h, w], dtype=torch.long) for _ in range(unit_count)]
        return [torch.tensor([1, 1, 1], dtype=torch.long) for _ in range(unit_count)]

    def _frame_visual_embeds_for_prompt(
        self,
        visual_embeds: torch.Tensor,
        local_positions: torch.Tensor,
    ) -> torch.Tensor:
        selected = visual_embeds.index_select(1, local_positions.to(device=visual_embeds.device, dtype=torch.long))
        return selected[0].detach().to(device="cpu", dtype=torch.bfloat16)

    def _grid_for_visual_token_count(self, token_count: int, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_count = max(int(token_count), 1)
        merge_size = max(int(getattr(self.owner, "merge_size", 2) or 2), 1)
        return torch.tensor(
            [1, merge_size, merge_size * token_count],
            dtype=torch.long,
            device=like.device if isinstance(like, torch.Tensor) else None,
        ).cpu()

    def _format_qwen3_timestamp(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        return f"<{seconds:.1f} seconds>"

    def _format_qwen3_cluster_timestamp(self, start_seconds: float, end_seconds: float) -> str:
        return self._format_qwen3_timestamp((float(start_seconds) + float(end_seconds)) / 2.0)

    def _build_append_local_causal_mask(
        self,
        past_frame_ids: Optional[torch.Tensor],
        current_frame_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        q_len = int(current_frame_ids.numel())
        if q_len <= 0:
            return None
        if past_frame_ids is None:
            past_frame_ids = current_frame_ids.new_empty((0,), dtype=torch.long)
        else:
            past_frame_ids = past_frame_ids.to(device=device, dtype=torch.long).view(-1)
        current_frame_ids = current_frame_ids.to(device=device, dtype=torch.long).view(-1)

        past_len = int(past_frame_ids.numel())
        all_frame_ids = torch.cat([past_frame_ids, current_frame_ids], dim=0)
        key_len = int(all_frame_ids.numel())
        q_idx = torch.arange(q_len, device=device, dtype=torch.long).view(q_len, 1)
        k_idx = torch.arange(key_len, device=device, dtype=torch.long).view(1, key_len)
        blocked = k_idx > (past_len + q_idx)

        window_units = self._local_window_units()
        if window_units > 0:
            q_frame = current_frame_ids.view(q_len, 1)
            k_frame = all_frame_ids.view(1, key_len)
            blocked = blocked | (k_frame < (q_frame - int(window_units) + 1))

        mask = torch.zeros((q_len, key_len), device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, torch.finfo(dtype).min)
        return mask.view(1, 1, q_len, key_len)

    def _build_append_flash_attention_groups(
        self,
        past_frame_ids: Optional[torch.Tensor],
        current_frame_ids: torch.Tensor,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Describe an exact unit-window append as contiguous Q/KV slices.

        Each tuple is ``(q_start, q_end, kv_start, kv_end)``. Consecutive query
        units with the same earliest visible key share one group. The queries
        are a suffix of their KV slice, so bottom-right causal FlashAttention
        has exactly the same visibility as the explicit unit mask.
        Irregular/non-monotonic layouts stay on the explicit-mask path.
        """

        current_ids = current_frame_ids.detach().view(-1).to(device="cpu", dtype=torch.long).tolist()
        if not current_ids:
            return None
        past_ids = (
            []
            if past_frame_ids is None
            else past_frame_ids.detach().view(-1).to(device="cpu", dtype=torch.long).tolist()
        )
        all_ids = [int(frame_id) for frame_id in past_ids + current_ids]
        if any(frame_id < 0 for frame_id in all_ids):
            return None
        if any(left > right for left, right in zip(all_ids, all_ids[1:])):
            return None

        groups: List[Tuple[int, int, int, int]] = []
        past_len = len(past_ids)
        window_units = self._local_window_units()
        q_start = 0
        while q_start < len(current_ids):
            frame_id = int(current_ids[q_start])
            q_end = q_start + 1
            while q_end < len(current_ids) and int(current_ids[q_end]) == frame_id:
                q_end += 1
            kv_start = bisect_left(all_ids, frame_id - window_units + 1)
            kv_end = past_len + q_end
            if groups and groups[-1][2] == kv_start:
                previous_q_start, _previous_q_end, previous_kv_start, _previous_kv_end = groups[-1]
                groups[-1] = (previous_q_start, q_end, previous_kv_start, kv_end)
            else:
                groups.append((q_start, q_end, kv_start, kv_end))
            q_start = q_end
        return groups

    def _build_append_attention_plan(
        self,
        past_frame_ids: Optional[torch.Tensor],
        current_frame_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[Tuple[int, int, int, int]]]]:
        if _full_kv_enabled(self.config):
            return None, None
        if not _as_bool(self.config, "streaming_lower_mask"):
            return None, None

        groups = self._build_append_flash_attention_groups(
            past_frame_ids=past_frame_ids,
            current_frame_ids=current_frame_ids,
        )
        if groups and self._uses_flash_attention_2():
            return None, groups

        if self._rekv_sink_enabled():
            if not self._uses_flash_attention_2():
                raise RuntimeError("ReKV sink streaming attention requires flash_attention_2")
            raise RuntimeError(
                "ReKV sink streaming attention requires a monotonic physical unit layout"
            )

        mask = self._build_append_local_causal_mask(
            past_frame_ids=past_frame_ids,
            current_frame_ids=current_frame_ids,
            device=device,
            dtype=dtype,
        )
        return mask, groups

    def _local_window_frames(self) -> int:
        value = _as_int(self.config, "shallow_prefill_local_window_frames")
        if value <= 0:
            raise ValueError("shallow_prefill_local_window_frames must be positive")
        return value

    def _local_window_units(self) -> int:
        temporal_patch_size = _as_int(self.config, "video_temporal_patch_size")
        if temporal_patch_size <= 0:
            raise ValueError("video_temporal_patch_size must be positive")
        frames = self._local_window_frames()
        return max(1, (frames + temporal_patch_size - 1) // temporal_patch_size)

    def _local_window_token_budget(self, unit_token_counts: Sequence[int]) -> int:
        counts = [int(count) for count in unit_token_counts]
        if not counts or any(count <= 0 for count in counts):
            raise ValueError("local-window unit token counts must be non-empty and positive")
        return sum(counts[-self._local_window_units() :])

    def _streaming_archive_enabled(self) -> bool:
        return _as_bool(self.config, "streaming_archive_full_history")

    def _streaming_archive_device(self) -> torch.device:
        value = str(self.config.get("streaming_archive_device", "cpu") or "cpu").strip().lower()
        if value not in {"cpu", "cuda"}:
            raise ValueError("streaming_archive_device must be 'cpu' or 'cuda'")
        return torch.device(value)

    def _commit_stream_lower_kv(
        self,
        session: _Qwen3VLStreamSession,
        *,
        appended_cache: Dict[int, Dict[str, torch.Tensor]],
        current_token_count: int,
        active_frame_ids: torch.Tensor,
    ) -> None:
        """Commit one append to the exact archive and bounded active cache."""

        active_frame_ids = active_frame_ids.detach().view(-1).long()
        current_token_count = int(current_token_count)
        if current_token_count < 0 or current_token_count > int(active_frame_ids.numel()):
            raise ValueError("current_token_count is inconsistent with active_frame_ids")

        if _full_kv_enabled(self.config):
            session.raw_lower_kv = appended_cache
            session.active_lower_kv = appended_cache
            session.token_frame_ids = active_frame_ids
            return

        archive_device = (
            self._streaming_archive_device()
            if self._streaming_archive_enabled()
            else next(iter(appended_cache.values()))["k"].device
        )
        for layer_idx, entry in appended_cache.items():
            key = entry["k"]
            value = entry["v"]
            positions = entry["positions"]
            current_start = int(key.shape[2]) - current_token_count
            current_key = key[:, :, current_start:, :].detach().to(archive_device)
            current_value = value[:, :, current_start:, :].detach().to(archive_device)
            current_positions = positions[..., current_start:].detach().to(archive_device)
            archived = session.raw_lower_kv.get(int(layer_idx))
            if archived is None:
                session.raw_lower_kv[int(layer_idx)] = {
                    "k": current_key,
                    "v": current_value,
                    "positions": current_positions,
                }
            else:
                session.raw_lower_kv[int(layer_idx)] = {
                    "k": torch.cat([archived["k"], current_key], dim=2),
                    "v": torch.cat([archived["v"], current_value], dim=2),
                    "positions": torch.cat([archived["positions"], current_positions], dim=-1),
                }

        window_units = self._local_window_units()
        if window_units > 0 and active_frame_ids.numel() > 0:
            newest_frame = int(active_frame_ids.max().item())
            keep = torch.nonzero(
                active_frame_ids >= newest_frame - window_units + 1,
                as_tuple=False,
            ).flatten()
        else:
            keep = torch.arange(active_frame_ids.numel(), device=active_frame_ids.device)

        keep_start = int(keep[0].item()) if keep.numel() > 0 else int(active_frame_ids.numel())
        expected_keep = torch.arange(
            keep_start,
            int(active_frame_ids.numel()),
            device=keep.device,
            dtype=keep.dtype,
        )
        if not torch.equal(keep, expected_keep):
            raise RuntimeError("active lower-KV window must be a contiguous suffix")

        active_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, entry in appended_cache.items():
            active_cache[int(layer_idx)] = {
                "k": entry["k"][:, :, keep_start:, :].detach(),
                "v": entry["v"][:, :, keep_start:, :].detach(),
                "positions": entry["positions"][..., keep_start:].detach(),
            }
        session.active_lower_kv = active_cache
        session.token_frame_ids = active_frame_ids[keep_start:].detach()

    def _build_text_inputs(
        self,
        prompt: str,
        *,
        include_answer_prefix: bool = False,
    ) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if include_answer_prefix:
            answer_prefix = getattr(self.owner, "assistant_answer_prefix", None)
            if answer_prefix is not None and not callable(answer_prefix):
                raise RuntimeError("owner.assistant_answer_prefix must be callable")
            text += str(answer_prefix() or "") if answer_prefix is not None else ""
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.owner._input_device())

    def _video_prompt_wrapper_token_ids(self, prompt: str, video_key: str) -> Tuple[List[int], List[int]]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen3 streaming internal_kv requires a tokenizer to build the video prompt wrapper.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_key)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer_prefix = getattr(self.owner, "assistant_answer_prefix", None)
        if answer_prefix is not None and not callable(answer_prefix):
            raise RuntimeError("owner.assistant_answer_prefix must be callable")
        text += str(answer_prefix() or "") if answer_prefix is not None else ""
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        video_positions = [idx for idx, token_id in enumerate(token_ids) if int(token_id) == int(self.owner.video_token_id)]
        if not video_positions:
            raise RuntimeError("No Qwen3 video placeholder token found in the streaming prompt wrapper.")
        first_video = int(video_positions[0])
        last_video = int(video_positions[-1])
        return token_ids[:first_video], token_ids[last_video + 1 :]

    def _embed_token_ids(
        self,
        token_ids: Sequence[int],
        embed_tokens,
        device: torch.device,
    ) -> torch.Tensor:
        input_ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
        return embed_tokens(input_ids)

    def _concat_lower_caches(
        self,
        left_cache: Dict[int, Dict[str, torch.Tensor]],
        right_cache: Dict[int, Dict[str, torch.Tensor]],
        lower_layer_ids: Sequence[int],
        *,
        consume_right_cache: bool = False,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        merged: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            left_k, left_v = self._raw_layer(left_cache, layer_idx)
            right_k, right_v = self._raw_layer(right_cache, layer_idx)
            left_positions = left_cache[layer_idx]["positions"].to(device=left_k.device, dtype=torch.long)
            right_positions = right_cache[layer_idx]["positions"].to(device=left_k.device, dtype=torch.long)
            merged[layer_idx] = {
                "k": torch.cat([left_k, right_k.to(device=left_k.device, dtype=left_k.dtype)], dim=2),
                "v": torch.cat([left_v, right_v.to(device=left_v.device, dtype=left_v.dtype)], dim=2),
                "positions": torch.cat([left_positions, right_positions], dim=-1),
            }
            if consume_right_cache:
                right_cache.pop(layer_idx, None)
        return merged

    def _initialize_stream_sink(
        self,
        session: _Qwen3VLStreamSession,
        *,
        input_embeds: torch.Tensor,
        language_positions: Optional[torch.Tensor],
        first_video_position: int,
        layers,
        rotary_emb,
        prune_layer: int,
    ) -> None:
        normalized_positions = self._normalize_rope_positions(
            language_positions,
            seq_len=int(input_embeds.shape[1]),
            device=input_embeds.device,
        )
        if session.visual_rope_base is None:
            first_visual = normalized_positions[..., int(first_video_position) : int(first_video_position) + 1]
            if int(first_visual.shape[-1]) != 1:
                raise RuntimeError("Qwen3-VL streaming prefill could not locate the first visual RoPE position")
            session.visual_rope_base = first_visual.detach().clone()
            session.next_visual_temporal_position = 0
        # Every sampled-video batch is rebuilt through the processor. Qwen may
        # insert batch-specific timestamp tokens before its visual placeholders,
        # so later batches need not report the same first-video position. Keep
        # the first batch's text prefix as the canonical reusable prompt prefix.
        if session.prompt_prefix_lower_kv:
            return
        prefix_len = max(0, int(first_video_position))
        if prefix_len <= 0:
            raise RuntimeError("Qwen3-VL streaming prefill requires a non-empty text prefix before video tokens")

        prefix_embeds = input_embeds[:, :prefix_len, :]
        prefix_positions = normalized_positions[..., :prefix_len]
        with torch.no_grad():
            prefix_hidden, prefix_cache = self._forward_lower_layers_raw(
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
        session.prompt_prefix_lower_kv = prefix_cache
        session.prompt_prefix_hidden_after_prune = prefix_hidden.detach()
        session.prompt_prefix_len = prefix_len
        if self._rekv_sink_enabled():
            session.sink_lower_kv = prefix_cache
            session.sink_len = prefix_len
            session.next_position = prefix_len

    def _append_stream_frames(
        self,
        session: _Qwen3VLStreamSession,
        sample: SampledVideo,
        video_path: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        frames = sample.frames
        if not frames:
            return

        # Preserve the original sampled RGB timeline before Qwen's temporal
        # video-unit coalescing can pad or otherwise group source frames.
        self._append_recent_source_frames(session, frames)

        if inputs is None:
            inputs = self._build_inputs(sample.video, sample.metadata, "", video_path)
        else:
            inputs = inputs.to(self.owner._input_device())
        layers, rotary_emb, _norm, _lm_head, _embed_tokens = self._language_parts()
        prune_layer = self._resolve_prune_layer(layers)
        lower_layer_ids = list(range(prune_layer))
        query_layer = prune_layer - 1

        input_ids = inputs["input_ids"]
        video_positions = self._video_positions(input_ids)
        if video_positions.numel() == 0:
            raise RuntimeError("No Qwen3 visual placeholder tokens were found while appending stream frames.")
        frames = self._coalesce_frames_to_video_units(frames, self._video_unit_count(inputs, len(frames)))
        real_frame_count = len(frames)
        frame_positions = self._split_video_positions(video_positions, len(frames), inputs.get("video_grid_thw"))
        unit_grid_rows = self._unit_grid_rows(inputs, real_frame_count)
        real_video_positions = torch.cat(frame_positions, dim=0)

        language = self._find_language_module()
        language_inputs = self._capture_prefill_language_inputs(inputs, language)
        input_embeds = language_inputs["inputs_embeds"]
        self._initialize_stream_sink(
            session,
            input_embeds=input_embeds,
            language_positions=language_inputs.get("position_ids"),
            first_video_position=int(video_positions[0].item()),
            layers=layers,
            rotary_emb=rotary_emb,
            prune_layer=prune_layer,
        )
        video_embeds = input_embeds.index_select(1, real_video_positions.to(input_embeds.device))
        visual_pos_masks = language_inputs.get("visual_pos_masks")
        deepstack_visual_embeds = language_inputs.get("deepstack_visual_embeds")
        current_deepstack = self._select_deepstack_visual_embeds(
            token_positions=real_video_positions,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        device = video_embeds.device
        batch_rope_positions = self._normalize_rope_positions(
            language_inputs.get("position_ids"),
            seq_len=int(input_embeds.shape[1]),
            device=device,
        )
        current_positions, batch_temporal_span = self._stream_prefill_visual_positions(
            session,
            batch_rope_positions,
            real_video_positions,
        )
        old_token_count = 0
        if _full_kv_enabled(self.config):
            old_token_count = sum(
                int(frame.token_indices.numel()) for frame in session.frame_states
            )
        elif session.hidden_after_prune is not None:
            old_token_count = int(session.hidden_after_prune.shape[1])

        current_frame_ids_parts: List[torch.Tensor] = []
        local_frame_indices: List[torch.Tensor] = []
        cursor = 0
        for offset, positions in enumerate(frame_positions):
            length = int(positions.numel())
            frame_token_indices = torch.arange(cursor, cursor + length, device=device, dtype=torch.long)
            local_frame_indices.append(frame_token_indices)
            current_frame_ids_parts.append(
                torch.full(
                    (length,),
                    int(session.next_frame_id + offset),
                    device=device,
                    dtype=torch.long,
                )
            )
            cursor += length
        current_frame_ids = torch.cat(current_frame_ids_parts, dim=0)
        historical_token_counts = [
            int(frame.token_indices.numel()) for frame in session.frame_states
        ] + [int(indices.numel()) for indices in local_frame_indices]
        local_window_tokens = (
            int(session.sink_len) + sum(historical_token_counts)
            if _full_kv_enabled(self.config)
            else self._local_window_token_budget(historical_token_counts)
        )
        append_mask, append_flash_groups = self._build_append_attention_plan(
            past_frame_ids=session.token_frame_ids,
            current_frame_ids=current_frame_ids,
            device=device,
            dtype=video_embeds.dtype,
        )
        current_visual_mask = torch.ones((1, int(video_embeds.shape[1])), device=device, dtype=torch.bool)
        past_lower_kv = session.active_lower_kv

        with torch.no_grad():
            new_hidden, new_raw_lower_kv = self._forward_lower_layers_raw(
                hidden_states=video_embeds,
                layers=layers,
                rotary_emb=rotary_emb,
                start_layer=0,
                end_layer=prune_layer,
                past_raw_kv=past_lower_kv,
                positions=current_positions,
                visual_pos_masks=current_visual_mask,
                deepstack_visual_embeds=current_deepstack,
                update_cache=True,
                capture_q_layers=None,
                attention_mask=append_mask,
                attention_unit_groups=append_flash_groups,
                sink_len=int(session.sink_len),
                sink_raw_kv=session.sink_lower_kv,
                local_window_tokens=local_window_tokens,
            )
        session.local_window_tokens = local_window_tokens

        active_frame_ids = (
            torch.cat(
                [
                    session.token_frame_ids.to(device=device, dtype=torch.long),
                    current_frame_ids,
                ],
                dim=0,
            )
            if session.token_frame_ids is not None
            else current_frame_ids
        )
        self._commit_stream_lower_kv(
            session,
            appended_cache=new_raw_lower_kv,
            current_token_count=int(video_embeds.shape[1]),
            active_frame_ids=active_frame_ids,
        )
        hidden_archive_device = (
            self._streaming_archive_device()
            if self._streaming_archive_enabled()
            else new_hidden.device
        )
        archived_new_hidden = new_hidden.detach().to(hidden_archive_device)
        if _full_kv_enabled(self.config):
            # The full-depth FullKV path only consumes the last prompt position
            # during decoding. Retain one exact visual hidden as a fallback and
            # derive logical visual length from frame token indices.
            session.hidden_after_prune = archived_new_hidden[:, -1:, :]
        elif session.hidden_after_prune is None:
            session.hidden_after_prune = archived_new_hidden
        else:
            session.hidden_after_prune = torch.cat(
                [
                    session.hidden_after_prune.to(
                        device=hidden_archive_device,
                        dtype=archived_new_hidden.dtype,
                    ),
                    archived_new_hidden,
                ],
                dim=1,
            )

        if _full_kv_enabled(self.config):
            session.visual_pos_masks = None
            session.deepstack_visual_embeds = None
        elif session.visual_pos_masks is None:
            session.visual_pos_masks = torch.ones(
                (1, int(session.hidden_after_prune.shape[1])),
                device=session.hidden_after_prune.device,
                dtype=torch.bool,
            )
        elif not _full_kv_enabled(self.config):
            old_mask = session.visual_pos_masks.to(device=session.hidden_after_prune.device, dtype=torch.bool)
            add_mask = torch.ones((1, int(new_hidden.shape[1])), device=old_mask.device, dtype=torch.bool)
            session.visual_pos_masks = torch.cat([old_mask, add_mask], dim=1)

        if not _full_kv_enabled(self.config) and self._selected_generate_mode() != "simple_prompt":
            session.deepstack_visual_embeds = self._concat_deepstack_embeds(
                session.deepstack_visual_embeds,
                current_deepstack,
                target_device=session.hidden_after_prune.device,
                target_dtype=session.hidden_after_prune.dtype,
            )

        key_for_query, _ = self._raw_layer(session.raw_lower_kv, query_layer)
        for local_idx, (frame, local_tokens) in enumerate(zip(frames[:real_frame_count], local_frame_indices)):
            token_indices = local_tokens + old_token_count
            frame_rope_positions = current_positions.index_select(-1, local_tokens.to(current_positions.device))
            frame_deepstack = (
                [
                    embeds.index_select(0, local_tokens.to(device=embeds.device, dtype=torch.long))
                    .detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    for embeds in current_deepstack
                ]
                if current_deepstack is not None and self._selected_generate_mode() == "simple_prompt"
                else None
            )
            session.frame_states.append(
                _FrameKVState(
                    frame_id=int(session.next_frame_id + local_idx),
                    sample_index=int(frame.index),
                    timestamp=float(frame.timestamp),
                    token_indices=token_indices.detach().clone(),
                    context_indices=token_indices.detach().clone(),
                    context_positions=frame_rope_positions.detach().clone(),
                    positions=frame_rope_positions.detach().clone(),
                    key_vec=self._normalized_key_vector(key_for_query, token_indices).detach(),
                    visual_embeds=self._frame_visual_embeds_for_prompt(video_embeds, local_tokens),
                    deepstack_embeds=frame_deepstack,
                    grid_thw=unit_grid_rows[local_idx].detach().clone() if local_idx < len(unit_grid_rows) else None,
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

        session.next_frame_id += int(real_frame_count)
        session.next_visual_temporal_position += max(batch_temporal_span, 0)
        session.next_position = max(session.next_position, int(current_positions.max().item()) + 1)
        session.last_timestamp = max(float(session.last_timestamp), max(float(frame.timestamp) for frame in frames[:real_frame_count]))
        if not self._streaming_archive_enabled() and not _full_kv_enabled(self.config):
            self._promote_stream_frames(session, lower_layer_ids)

    def _stitch_stream_visual_rope_positions(
        self,
        session: _Qwen3VLStreamSession,
        batch_rope_positions: torch.Tensor,
        visual_token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Join processor batches as slices of one video in Qwen's T/H/W RoPE space."""
        local = self._select_rope_positions(batch_rope_positions, visual_token_indices)
        if session.visual_rope_base is None:
            raise RuntimeError("Qwen3-VL streaming visual RoPE base was not initialized")
        relative = local - local[:, :1]
        local_temporal_span = int(relative[0].max().item()) + 1
        relative[0] += int(session.next_visual_temporal_position)
        stitched = relative + session.visual_rope_base.to(device=local.device, dtype=torch.long)
        return stitched, local_temporal_span

    def _stream_prefill_visual_positions(
        self,
        session: _Qwen3VLStreamSession,
        batch_rope_positions: torch.Tensor,
        visual_token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        if self._rope_position_mode() == "relative":
            return (
                self._make_scalar_positions(
                    session.next_position,
                    int(visual_token_indices.numel()),
                    batch_rope_positions.device,
                ),
                0,
            )
        return self._stitch_stream_visual_rope_positions(
            session,
            batch_rope_positions,
            visual_token_indices,
        )

    def _concat_deepstack_embeds(
        self,
        old_embeds: Optional[List[torch.Tensor]],
        new_embeds: Optional[List[torch.Tensor]],
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[List[torch.Tensor]]:
        if old_embeds is None:
            if new_embeds is None:
                return None
            return [emb.detach().to(device=target_device, dtype=target_dtype) for emb in new_embeds]
        if new_embeds is None or len(old_embeds) != len(new_embeds):
            return None
        merged: List[torch.Tensor] = []
        for old, new in zip(old_embeds, new_embeds):
            merged.append(
                torch.cat(
                    [
                        old.detach().to(device=target_device, dtype=target_dtype),
                        new.detach().to(device=target_device, dtype=target_dtype),
                    ],
                    dim=0,
                )
            )
        return merged

    def _merge_frame_into_stream_clusters(
        self,
        session: _Qwen3VLStreamSession,
        frame: _FrameKVState,
        lower_layer_ids: Sequence[int],
    ) -> None:
        if not _as_bool(self.config, "long_cluster_enabled"):
            return
        if session.hidden_after_prune is None:
            return
        cluster_layer_ids = list(lower_layer_ids)
        if self._selected_generate_mode() == "simple_prompt":
            if not cluster_layer_ids:
                raise RuntimeError("Qwen3-VL long-cluster retrieval requires a lower query layer")
            cluster_layer_ids = [cluster_layer_ids[-1]]
        frame_lower_kv = self._extract_frame_lower_kv(
            session.raw_lower_kv,
            cluster_layer_ids,
            frame.token_indices,
        )
        frame_hidden = self._index_hidden_positions(session.hidden_after_prune, frame.token_indices).detach()
        frame_deepstack = frame.deepstack_embeds
        if frame_deepstack is None:
            frame_deepstack = self._select_deepstack_visual_embeds(
                token_positions=frame.token_indices,
                visual_pos_masks=session.visual_pos_masks,
                deepstack_visual_embeds=session.deepstack_visual_embeds,
            )
        if not session.clusters:
            session.clusters.append(_LongKVCluster.from_frame(frame, frame_lower_kv, frame_hidden, frame_deepstack))
            return

        latest = session.clusters[-1]
        similarity = float(torch.dot(latest.key_vec.float(), frame.key_vec.float()).item())
        if similarity < _as_float(self.config, "cluster_threshold"):
            session.clusters.append(_LongKVCluster.from_frame(frame, frame_lower_kv, frame_hidden, frame_deepstack))
        else:
            latest.merge(frame, frame_lower_kv, frame_hidden, frame_deepstack)

    def _promote_stream_frames(
        self,
        session: _Qwen3VLStreamSession,
        lower_layer_ids: Sequence[int],
    ) -> None:
        detail_capacity = max(self._retrieval_search_window_units(), self._retrieval_recent_units(), 0)
        promote_count = max(0, len(session.frame_states) - detail_capacity)
        if promote_count <= 0:
            return
        if session.hidden_after_prune is None:
            return

        promote_frames = list(session.frame_states[:promote_count])
        keep_frames = list(session.frame_states[promote_count:])
        if _as_bool(self.config, "long_cluster_enabled"):
            for frame in promote_frames:
                self._merge_frame_into_stream_clusters(session, frame, lower_layer_ids)

        if keep_frames:
            keep_positions = torch.cat(
                [frame.context_indices.to(device=session.hidden_after_prune.device, dtype=torch.long) for frame in keep_frames],
                dim=0,
            )
            keep_positions = torch.unique(keep_positions, sorted=True)
        else:
            keep_positions = torch.empty((0,), device=session.hidden_after_prune.device, dtype=torch.long)

        old_len = int(session.hidden_after_prune.shape[1])
        old_to_new = torch.full((old_len,), -1, device=session.hidden_after_prune.device, dtype=torch.long)
        if keep_positions.numel() > 0:
            old_to_new[keep_positions] = torch.arange(keep_positions.numel(), device=session.hidden_after_prune.device)

        compact_raw: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, entry in session.raw_lower_kv.items():
            key = entry["k"]
            value = entry["v"]
            positions = entry.get("positions")
            gather_k = keep_positions.to(device=key.device, dtype=torch.long)
            gather_v = keep_positions.to(device=value.device, dtype=torch.long)
            compact_entry = {
                "k": key.index_select(2, gather_k).detach(),
                "v": value.index_select(2, gather_v).detach(),
            }
            if isinstance(positions, torch.Tensor):
                compact_entry["positions"] = positions.index_select(-1, keep_positions.to(device=positions.device, dtype=torch.long)).detach()
            session_q = entry.get("q")
            if isinstance(session_q, torch.Tensor):
                compact_entry["q"] = session_q[:, :, :0, :].detach()
            compact_raw[int(layer_idx)] = compact_entry
        session.raw_lower_kv = compact_raw
        session.hidden_after_prune = session.hidden_after_prune.index_select(1, keep_positions).detach()
        if session.visual_pos_masks is not None:
            session.visual_pos_masks = session.visual_pos_masks.to(device=keep_positions.device).index_select(1, keep_positions).detach()
        if isinstance(session.deepstack_visual_embeds, list):
            compact_deepstack: List[torch.Tensor] = []
            for embeds in session.deepstack_visual_embeds:
                compact_deepstack.append(embeds.index_select(0, keep_positions.to(device=embeds.device, dtype=torch.long)).detach())
            session.deepstack_visual_embeds = compact_deepstack

        session.frame_states = [self._remap_frame_token_indices(frame, old_to_new) for frame in keep_frames]

    def _query_vector_from_current_q(
        self,
        raw_lower_kv: Dict[int, Dict[str, torch.Tensor]],
        query_layer: int,
    ) -> torch.Tensor:
        layer_cache = raw_lower_kv.get(query_layer)
        if not isinstance(layer_cache, dict) or not isinstance(layer_cache.get("q"), torch.Tensor):
            raise RuntimeError(f"Missing raw Q capture for streaming retrieval query at lower layer {query_layer}.")
        query = layer_cache["q"]
        key, _ = self._raw_layer(raw_lower_kv, query_layer)
        positions = torch.arange(int(query.shape[2]), device=query.device, dtype=torch.long)
        return self._normalized_query_vector(query, positions, key_head_count=int(key.shape[1])).detach()

    def _select_stream_memory(
        self,
        session: _Qwen3VLStreamSession,
        query_vec: torch.Tensor,
        query_layer: int,
        prompt: str,
        probe_query_vecs: Optional[Dict[str, torch.Tensor]] = None,
        attention_distribution_layer: Optional[int] = None,
        attention_distribution_features: Optional[Dict[str, float]] = None,
        attention_distribution_observation_features: Optional[Dict[int, Dict[str, float]]] = None,
        token_vote_query_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        token_vote_query_indices: Optional[Sequence[int]] = None,
        rotary_emb=None,
        layers: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        sync_latency = _as_bool(self.config, "latency_sync_cuda")
        if sync_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        retrieval_started = time.perf_counter()
        phase_timing = getattr(self, "_last_query_phase_timing", None)
        if isinstance(phase_timing, dict):
            phase_timing["retrieval_selection_started_time"] = retrieval_started
        frames = list(session.frame_states)
        total_frames = len(frames)
        recent_n = self._retrieval_recent_units()
        topk = self._retrieval_topk_units()
        search_n = self._retrieval_search_window_units()

        recent_start = max(total_frames - recent_n, 0) if recent_n > 0 else total_frames
        search_start = max(total_frames - search_n, 0) if search_n > 0 else total_frames
        search_end = max(search_start, recent_start)
        search_candidates = frames[search_start:recent_start] if search_start < recent_start else []
        recent_frames = frames[recent_start:]
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
                if token_vote_query_cache is None or rotary_emb is None or layers is None:
                    raise RuntimeError(
                        "streaming shallow-layer token voting requires query capture, "
                        "model layers, and RoPE"
                    )
                layer_indices = list(
                    range(self._retrieval_vote_layer_start(), int(query_layer) + 1)
                )
                retrieved_seed, scored_short, vote_stats = (
                    self._select_shallow_layer_token_vote_frames(
                        candidate_frames=search_candidates,
                        q_cache=token_vote_query_cache,
                        k_cache=session.raw_lower_kv,
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
            detail_candidates = frames[search_start:]
            candidate_by_id = {frame.frame_id: frame for frame in detail_candidates}
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

        selected_short = sorted(
            {frame.frame_id: frame for frame in retrieved + recent_frames}.values(),
            key=lambda frame: frame.frame_id,
        )
        token_selection_stats: Dict[str, Any] = {"retrieval_selection_granularity": "unit"}
        token_selection_stats.update(vote_stats)
        if self._retrieval_selection_granularity() == "token":
            key, _ = self._raw_layer(session.raw_lower_kv, query_layer)
            token_retrieved, selected_short, token_selection_stats = self._select_token_level_short_frames(
                candidate_frames=search_candidates,
                selected_unit_frames=selected_short,
                recent_frames=recent_frames,
                key=key,
                query_vec=query_vec,
            )
            retrieved = token_retrieved
        selected_clusters = self._retrieve_long_kv(session.clusters, query_vec)
        selection = {
            "query_vec": query_vec,
            "short": selected_short,
            "clusters": selected_clusters,
            "all_clusters": list(session.clusters),
            "retrieved": retrieved,
            "retrieved_seed": retrieved_seed,
            "retrieval_reference_expanded_unit_ids": retrieval_reference_expanded_unit_ids,
            "recent": recent_frames,
            "search_candidates": search_candidates,
            "short_scores": scored_short,
            "search_start": search_start,
            "search_end": search_end,
            "token_selection_stats": token_selection_stats,
            "probe_query_vecs": dict(probe_query_vecs or {}),
            "attention_distribution_layer": attention_distribution_layer,
            "attention_distribution_features": attention_distribution_features,
            "attention_distribution_observation_features": attention_distribution_observation_features,
        }
        if recent_unit_score_observation is not None:
            selection["recent_retrieval_unit_scores"] = recent_unit_score_observation
        if sync_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        retrieval_finished = time.perf_counter()
        if isinstance(phase_timing, dict):
            phase_timing["retrieval_selection_finished_time"] = retrieval_finished
            phase_timing["query_logit_gate_started_time"] = retrieval_finished
        gated_selection = self._apply_task_gate(
            selection,
            prompt=prompt,
            source="stream",
        )
        if sync_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        if isinstance(phase_timing, dict):
            phase_timing["query_logit_gate_finished_time"] = time.perf_counter()
        return self._attach_recent_source_frames(session, gated_selection)

    def _compose_stream_visual_positions(
        self,
        clusters: Sequence[_LongKVCluster],
        short_frames: Sequence[_FrameKVState],
        selected_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._rope_position_mode() == "relative":
            return self._make_scalar_positions(0, selected_len, device)
        parts: List[torch.Tensor] = []
        for cluster in clusters:
            parts.append(cluster.positions.round().long().to(device))
        for frame in short_frames:
            parts.append(frame.positions.long().to(device))
        if not parts:
            return self._make_scalar_positions(0, 0, device)
        positions = torch.cat(parts, dim=-1)
        if int(positions.shape[-1]) != int(selected_len):
            raise RuntimeError("stream selected visual position count does not match selected visual hidden length.")
        return positions

    def _build_stream_selected_visual_cache(
        self,
        session: _Qwen3VLStreamSession,
        selection: Dict[str, Any],
        lower_layer_ids: Sequence[int],
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Dict[str, Any],
    ]:
        if session.hidden_after_prune is None:
            device = self.owner._input_device()
            hidden_size = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)
            empty_hidden = torch.empty((1, 0, hidden_size), device=device)
            return {}, empty_hidden, self._make_scalar_positions(0, 0, device), None, None, {}

        if _full_kv_enabled(self.config):
            if session.clusters:
                raise RuntimeError("FullKV Qwen3-VL cannot rebuild context from clusters")
            expected_tokens = sum(
                int(frame.token_indices.numel()) for frame in session.frame_states
            )
            if expected_tokens <= 0:
                raise RuntimeError("FullKV Qwen3-VL frame/token archive is empty")
            visual_len = expected_tokens
            visual_hidden = session.hidden_after_prune[:, -1:, :]
            visual_positions = self._make_scalar_positions(
                0,
                visual_len,
                visual_hidden.device,
            )
            visual_cache: Dict[int, Dict[str, torch.Tensor]] = {}
            for layer_idx in lower_layer_ids:
                base_k, base_v = self._raw_layer(session.raw_lower_kv, layer_idx)
                if int(base_k.shape[2]) != visual_len or int(base_v.shape[2]) != visual_len:
                    raise RuntimeError(
                        "FullKV Qwen3-VL layer archive is incomplete: "
                        f"layer={layer_idx}, k={int(base_k.shape[2])}, "
                        f"v={int(base_v.shape[2])}, expected={visual_len}"
                    )
                visual_cache[int(layer_idx)] = {
                    "k": base_k,
                    "v": base_v,
                    "positions": visual_positions.to(device=base_k.device, dtype=torch.long),
                }
            visual_mask = torch.ones(
                (1, 1),
                device=visual_hidden.device,
                dtype=torch.bool,
            )
            stats = {
                "stream_mode": "open_window",
                "memory_policy": "full_kv",
                "stream_video_key": session.video_key,
                "stream_last_timestamp": float(session.last_timestamp),
                "stream_detail_units": len(session.frame_states),
                "stream_cluster_count": 0,
                "stream_selected_short_units": [
                    frame.frame_id for frame in session.frame_states
                ],
                "stream_selected_visual_tokens": visual_len,
                "use_rekv_sink": _as_bool(self.config, "use_rekv_sink"),
                "rekv_sink_len": int(session.sink_len),
                "shallow_prefill_local_window_frames": None,
                "local_window_units": None,
                "local_window_tokens": int(session.local_window_tokens),
            }
            return (
                visual_cache,
                visual_hidden,
                visual_positions,
                visual_mask,
                None,
                stats,
            )

        short_frames: List[_FrameKVState] = list(selection["short"])
        clusters: List[_LongKVCluster] = list(selection["clusters"])
        deepstack_count = len(session.deepstack_visual_embeds) if isinstance(session.deepstack_visual_embeds, list) else 0

        hidden_parts: List[torch.Tensor] = []
        deepstack_parts: List[List[torch.Tensor]] = [[] for _ in range(deepstack_count)]
        deepstack_complete = deepstack_count > 0

        def _append_visual(hidden: torch.Tensor, deepstack_embeds: Optional[List[torch.Tensor]]) -> None:
            nonlocal deepstack_complete
            hidden_parts.append(hidden.to(device=session.hidden_after_prune.device, dtype=session.hidden_after_prune.dtype))
            if not deepstack_complete:
                return
            if deepstack_embeds is None or len(deepstack_embeds) < deepstack_count:
                deepstack_complete = False
                return
            for layer_idx in range(deepstack_count):
                deepstack_parts[layer_idx].append(
                    deepstack_embeds[layer_idx].to(
                        device=session.hidden_after_prune.device,
                        dtype=session.hidden_after_prune.dtype,
                    )
                )

        for cluster in clusters:
            _append_visual(cluster.hidden, cluster.deepstack_embeds)
        for frame in short_frames:
            frame_hidden = self._index_hidden_positions(session.hidden_after_prune, frame.token_indices).detach()
            frame_deepstack = self._select_deepstack_visual_embeds(
                token_positions=frame.token_indices,
                visual_pos_masks=session.visual_pos_masks,
                deepstack_visual_embeds=session.deepstack_visual_embeds,
            )
            _append_visual(frame_hidden, frame_deepstack)

        if hidden_parts:
            visual_hidden = torch.cat(hidden_parts, dim=1)
        else:
            visual_hidden = session.hidden_after_prune[:, :0, :]
        visual_positions = self._compose_stream_visual_positions(
            clusters=clusters,
            short_frames=short_frames,
            selected_len=int(visual_hidden.shape[1]),
            device=visual_hidden.device,
        )
        visual_mask = torch.ones((1, int(visual_hidden.shape[1])), device=visual_hidden.device, dtype=torch.bool)

        selected_deepstack: Optional[List[torch.Tensor]] = None
        if deepstack_complete:
            selected_deepstack = []
            for layer_idx in range(deepstack_count):
                if deepstack_parts[layer_idx]:
                    selected_deepstack.append(torch.cat(deepstack_parts[layer_idx], dim=0))
                else:
                    selected_deepstack.append(session.hidden_after_prune.new_empty((0, int(session.hidden_after_prune.shape[-1]))))

        visual_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            k_parts: List[torch.Tensor] = []
            v_parts: List[torch.Tensor] = []
            base_k, base_v = self._raw_layer(session.raw_lower_kv, layer_idx)
            for cluster in clusters:
                k, v = cluster.lower_kv[layer_idx]
                k_parts.append(k.to(device=base_k.device, dtype=base_k.dtype))
                v_parts.append(v.to(device=base_v.device, dtype=base_v.dtype))
            for frame in short_frames:
                k_parts.append(self._index_cache_positions(base_k, frame.token_indices))
                v_parts.append(self._index_cache_positions(base_v, frame.token_indices))
            if k_parts:
                raw_key = torch.cat(k_parts, dim=2)
                value = torch.cat(v_parts, dim=2)
            else:
                raw_key = base_k[:, :, :0, :]
                value = base_v[:, :, :0, :]
            visual_cache[layer_idx] = {
                "k": raw_key,
                "v": value,
                "positions": visual_positions.to(device=raw_key.device, dtype=torch.long),
            }

        stats = {
            "stream_mode": "open_window",
            "stream_video_key": session.video_key,
            "stream_last_timestamp": float(session.last_timestamp),
            "stream_detail_units": len(session.frame_states),
            "stream_cluster_count": len(session.clusters),
            "stream_cluster_unit_counts": [cluster.count for cluster in session.clusters],
            "stream_selected_long_clusters": len(clusters),
            "stream_selected_cluster_unit_counts": [cluster.count for cluster in clusters],
            "stream_search_candidates": len(selection["search_candidates"]),
            "stream_retrieved_seed_units": [frame.frame_id for frame in selection["retrieved_seed"]],
            "stream_retrieved_short_units": [frame.frame_id for frame in selection["retrieved"]],
            "stream_recent_units": [frame.frame_id for frame in selection["recent"]],
            "stream_selected_short_units": [frame.frame_id for frame in short_frames],
            "stream_selected_visual_tokens": int(visual_hidden.shape[1]),
            "use_rekv_sink": _as_bool(self.config, "use_rekv_sink"),
            "rekv_sink_len": int(session.sink_len),
            "shallow_prefill_local_window_frames": self._local_window_frames(),
            "local_window_units": self._local_window_units(),
            "local_window_tokens": int(session.local_window_tokens),
        }
        stats.update(selection.get("token_selection_stats") or {})
        return visual_cache, visual_hidden, visual_positions, visual_mask, selected_deepstack, stats

    def _offset_stream_visual_context(
        self,
        visual_cache: Dict[int, Dict[str, torch.Tensor]],
        visual_positions: torch.Tensor,
        prefix_len: int,
        lower_layer_ids: Sequence[int],
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor]:
        offset_positions = visual_positions + int(prefix_len)
        offset_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in lower_layer_ids:
            entry = visual_cache[layer_idx]
            offset_cache[layer_idx] = {
                "k": entry["k"],
                "v": entry["v"],
                "positions": offset_positions.to(device=entry["k"].device, dtype=torch.long),
            }
        return offset_cache, offset_positions
