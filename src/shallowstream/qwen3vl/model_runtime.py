"""Qwen3VLModelRuntimeMixin implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import _as_bool, _as_int, _get_active_attn_implementation
from .frame_memory import SampledFrame, _l2_normalize
from .state import _StopAfterInputEmbeds


class Qwen3VLModelRuntimeMixin:

    def _video_unit_count(self, inputs: Dict[str, Any], fallback: int) -> int:
        grid = inputs.get("video_grid_thw")
        if isinstance(grid, torch.Tensor) and grid.numel() > 0:
            grid = grid.detach().cpu()
            if grid.dim() == 2 and int(grid.shape[0]) >= 1 and int(grid.shape[1]) >= 1:
                return max(1, int(grid[0, 0].item()))
        return max(1, int(fallback))

    def _coalesce_frames_to_video_units(
        self,
        frames: Sequence[SampledFrame],
        unit_count: int,
    ) -> List[SampledFrame]:
        if unit_count <= 0 or not frames:
            return []
        if len(frames) == unit_count:
            return [
                SampledFrame(
                    index=frame.index,
                    timestamp=frame.timestamp,
                    image=frame.image,
                    embedding=frame.embedding,
                    source_frames=list(getattr(frame, "source_frames", None) or [frame]),
                )
                for frame in frames
            ]

        groups = np.array_split(np.arange(len(frames)), unit_count)
        units: List[SampledFrame] = []
        for group in groups:
            if len(group) == 0:
                continue
            group_frames = [frames[int(i)] for i in group]
            mid = group_frames[len(group_frames) // 2]
            embeddings = [frame.embedding for frame in group_frames if frame.embedding.size > 0]
            embedding = (
                _l2_normalize(np.mean(embeddings, axis=0))
                if len(embeddings) == len(group_frames)
                else np.empty((0,), dtype=np.float32)
            )
            units.append(
                SampledFrame(
                    index=int(group_frames[0].index),
                    timestamp=float(np.mean([frame.timestamp for frame in group_frames])),
                    image=mid.image,
                    embedding=embedding,
                    source_frames=[
                        source
                        for frame in group_frames
                        for source in (getattr(frame, "source_frames", None) or [frame])
                    ],
                )
            )
        return units

    def _prepare_images(self, frames: Sequence[SampledFrame]) -> Tuple[List[Image.Image], int]:
        images = [frame.image.convert("RGB") for frame in frames]
        return images, len(images)

    def _build_inputs(
        self,
        video: torch.Tensor,
        video_metadata: Dict[str, Any],
        prompt: str,
        video_path: str,
    ) -> Dict[str, Any]:
        inputs = self._build_inputs_cpu(video, video_metadata, prompt, video_path)
        return inputs.to(self.owner._input_device())

    def _build_inputs_cpu(
        self,
        video: torch.Tensor,
        video_metadata: Dict[str, Any],
        prompt: str,
        video_path: str,
        *,
        processor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        active_processor = processor if processor is not None else self.processor
        processor_video_metadata = {
            key: value
            for key, value in video_metadata.items()
            if not str(key).startswith("stream_")
        }
        content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ]
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = active_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return active_processor(
            text=[text],
            videos=[video],
            video_metadata=[processor_video_metadata],
            padding=True,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )

    def _capture_prefill_language_inputs(self, inputs: Dict[str, Any], language) -> Dict[str, Any]:
        captured: Dict[str, Any] = {}

        def _clone_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, list):
                return [_clone_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_clone_value(item) for item in value)
            return value

        def _save_kwargs(kwargs: Dict[str, Any]) -> None:
            for key in (
                "inputs_embeds",
                "position_ids",
                "attention_mask",
                "visual_pos_masks",
                "deepstack_visual_embeds",
            ):
                if key in kwargs:
                    captured[key] = _clone_value(kwargs[key])
            raise _StopAfterInputEmbeds()

        def pre_hook_with_kwargs(_module, args, kwargs):
            if not isinstance(kwargs, dict) or not isinstance(kwargs.get("inputs_embeds"), torch.Tensor):
                raise RuntimeError("Could not capture Qwen3 language_model inputs.")
            _save_kwargs(kwargs)

        def pre_hook_no_kwargs(_module, args):
            raise RuntimeError("Qwen3 language_model inputs must be captured with keyword-aware hooks.")

        try:
            handle = language.register_forward_pre_hook(pre_hook_with_kwargs, with_kwargs=True)
        except TypeError:
            handle = language.register_forward_pre_hook(pre_hook_no_kwargs)

        try:
            with torch.no_grad():
                self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        except _StopAfterInputEmbeds:
            pass
        finally:
            handle.remove()

        if not isinstance(captured.get("inputs_embeds"), torch.Tensor):
            raise RuntimeError("Qwen3 language_model inputs_embeds were not captured.")
        return captured

    def _language_parts(self):
        language = self._find_language_module()
        layers = getattr(language, "layers")
        rotary_emb = self._find_attr("rotary_emb", language)
        norm = self._find_attr("norm", language)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            lm_head = self._find_attr("lm_head", self.model)
        embed_tokens = getattr(language, "embed_tokens", None)
        if embed_tokens is None:
            embed_tokens = self.model.get_input_embeddings()
        if rotary_emb is None or norm is None or lm_head is None or embed_tokens is None:
            raise RuntimeError("Could not locate Qwen3 language layers/rotary/norm/lm_head/embed_tokens.")
        return layers, rotary_emb, norm, lm_head, embed_tokens

    def _find_language_module(self):
        candidates = [
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(self.model, "model", None),
        ]
        for module in candidates:
            if module is not None and hasattr(module, "layers"):
                return module
        raise RuntimeError("Could not locate Qwen3 language_model.layers.")

    def _find_attr(self, name: str, root: Any) -> Any:
        visited: set[int] = set()
        queue = [root]
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in visited:
                continue
            visited.add(id(module))
            if hasattr(module, name):
                return getattr(module, name)
            for child_name in ("model", "language_model"):
                child = getattr(module, child_name, None)
                if child is not None:
                    queue.append(child)
        return None

    def _video_token_id(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("video_token_id", "video_token_index"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            if token_id is not None and token_id != tokenizer.unk_token_id:
                return int(token_id)
        return 151656

    def _video_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_id = self._video_token_id()
        return torch.nonzero(input_ids[0] == token_id, as_tuple=False).flatten()

    def _split_video_positions(
        self,
        video_positions: torch.Tensor,
        frame_count: int,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if frame_count <= 0:
            return []
        if int(video_positions.numel()) < frame_count:
            raise RuntimeError(
                f"visual token count {int(video_positions.numel())} is smaller than frame count {frame_count}."
            )

        if isinstance(grid_thw, torch.Tensor) and int(grid_thw.numel()) > 0:
            grid = grid_thw.detach().to(device=video_positions.device, dtype=torch.long)
            if grid.dim() == 2 and int(grid.shape[1]) >= 3:
                merge_size = int(
                    getattr(
                        self.owner,
                        "merge_size",
                        getattr(getattr(self.processor, "image_processor", None), "merge_size", 2),
                    )
                    or 2
                )
                merge_area = max(1, merge_size * merge_size)
                counts: List[int] = []
                if int(grid.shape[0]) == 1 and int(grid[0, 0].item()) == frame_count:
                    frame_seqlen = max(1, int(grid[0, 1].item() * grid[0, 2].item()) // merge_area)
                    counts = [frame_seqlen for _ in range(frame_count)]
                elif int(grid.shape[0]) >= frame_count:
                    counts = [
                        max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
                        for row in grid[:frame_count]
                    ]
                expected = sum(counts)
                if counts and expected <= int(video_positions.numel()):
                    chunks: List[torch.Tensor] = []
                    cursor = 0
                    for count in counts:
                        chunks.append(video_positions[cursor : cursor + count].contiguous())
                        cursor += count
                    return chunks

        chunks = torch.tensor_split(video_positions, frame_count)
        return [chunk.contiguous() for chunk in chunks]

    def _split_video_context_positions(
        self,
        frame_positions: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Attach inter-video text, including Qwen3 timestamp tokens, to units."""

        context_positions: List[torch.Tensor] = []
        previous_end: Optional[int] = None
        for positions in frame_positions:
            if positions.numel() == 0:
                context_positions.append(positions.contiguous())
                continue
            current = positions.detach().long()
            if previous_end is None:
                context_positions.append(current.contiguous())
            else:
                start = int(previous_end) + 1
                end = int(current[-1].item()) + 1
                if start < end:
                    context_positions.append(
                        torch.arange(start, end, device=current.device, dtype=torch.long)
                    )
                else:
                    context_positions.append(current.contiguous())
            previous_end = int(current[-1].item())
        return context_positions

    def _raw_layer(self, raw_lower_kv: Dict[int, Dict[str, torch.Tensor]], layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = raw_lower_kv[layer_idx]
        return entry["k"], entry["v"]

    def _normalize_rope_positions(
        self,
        position_ids: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is None:
            base = torch.arange(seq_len, device=device, dtype=torch.long)
            return base.view(1, -1).expand(3, -1).contiguous()

        pos = position_ids.detach().to(device=device, dtype=torch.long)
        if pos.dim() == 3:
            # HF Qwen3-VL can pass either (4, B, L): text+T/H/W, or
            # (3, B, L): T/H/W.  Lower attention RoPE uses only T/H/W.
            if int(pos.shape[0]) == 4:
                pos = pos[1:]
            elif int(pos.shape[0]) != 3:
                raise RuntimeError(f"Unsupported Qwen3 position_ids shape: {tuple(pos.shape)}")
            if int(pos.shape[1]) != 1:
                raise RuntimeError("ShallowStream_Qwen3VL_V3 currently expects batch_size=1.")
            pos = pos[:, 0, :]
        elif pos.dim() == 2:
            if int(pos.shape[0]) == 4:
                pos = pos[1:]
            elif int(pos.shape[0]) == 3:
                pass
            elif int(pos.shape[0]) == 1:
                pos = pos.expand(3, -1)
            else:
                pos = pos[0:1, :].expand(3, -1)
        elif pos.dim() == 1:
            pos = pos.view(1, -1).expand(3, -1)
        else:
            raise RuntimeError(f"Unsupported Qwen3 position_ids shape: {tuple(pos.shape)}")

        if int(pos.shape[-1]) != int(seq_len):
            raise RuntimeError(f"position length {int(pos.shape[-1])} != sequence length {int(seq_len)}")
        return pos.contiguous()

    def _select_rope_positions(self, rope_positions: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        token_indices = token_indices.to(device=rope_positions.device, dtype=torch.long)
        if rope_positions.dim() == 1:
            return rope_positions.index_select(0, token_indices)
        return rope_positions.index_select(-1, token_indices)

    def _build_streaming_local_causal_mask(
        self,
        seq_len: int,
        frame_positions: Sequence[torch.Tensor],
        video_positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        window_units = self._local_window_units()
        if window_units <= 0 or not frame_positions:
            return None

        token_frame = torch.full((seq_len,), -1, device=device, dtype=torch.long)
        for frame_id, positions in enumerate(frame_positions):
            if positions.numel() == 0:
                continue
            pos = positions.to(device=device, dtype=torch.long)
            valid = pos[(pos >= 0) & (pos < seq_len)]
            if valid.numel() > 0:
                token_frame[valid] = int(frame_id)
        if not torch.any(token_frame >= 0):
            return None

        q_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(seq_len, 1)
        k_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
        blocked = k_idx > q_idx

        q_frame = token_frame.view(seq_len, 1)
        k_frame = token_frame.view(1, seq_len)
        key_is_video = k_frame >= 0

        query_is_video = q_frame >= 0
        min_visible_frame = q_frame - int(window_units) + 1
        blocked = blocked | (query_is_video & key_is_video & (k_frame < min_visible_frame))

        last_video_pos = int(video_positions.max().item())
        suffix_query = q_idx > last_video_pos
        last_window_start = max(0, len(frame_positions) - int(window_units))
        blocked = blocked | (suffix_query & key_is_video & (k_frame < last_window_start))

        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, torch.finfo(dtype).min)
        return mask.view(1, 1, seq_len, seq_len)

    def _build_full_prefill_attention_groups(
        self,
        *,
        seq_len: int,
        frame_positions: Sequence[torch.Tensor],
    ) -> List[Tuple[int, int, int, int]]:
        """Describe full-prompt sink/local attention as physical Q/KV slices."""

        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        spans: List[Tuple[int, int]] = []
        for positions in frame_positions:
            values = sorted({int(value) for value in positions.detach().cpu().view(-1).tolist()})
            if not values or values != list(range(values[0], values[-1] + 1)):
                raise RuntimeError("Qwen full-prefill visual units must be non-empty contiguous spans")
            spans.append((values[0], values[-1] + 1))
        if not spans:
            return [(0, seq_len, 0, seq_len)]
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            raise RuntimeError("Qwen full-prefill visual unit spans must be contiguous")

        groups: List[Tuple[int, int, int, int]] = []
        prefix_end = spans[0][0]
        if prefix_end > 0:
            groups.append((0, prefix_end, 0, prefix_end))

        window_units = self._local_window_units()
        for frame_id, (q_start, q_end) in enumerate(spans):
            first_visible = max(0, frame_id - window_units + 1)
            kv_start = spans[first_visible][0]
            if groups and groups[-1][2] == kv_start and groups[-1][1] == q_start:
                previous_q_start, _previous_q_end, previous_kv_start, _previous_kv_end = groups[-1]
                groups[-1] = (previous_q_start, q_end, previous_kv_start, q_end)
            else:
                groups.append((q_start, q_end, kv_start, q_end))

        suffix_start = spans[-1][1]
        if suffix_start < seq_len:
            first_visible = max(0, len(spans) - window_units)
            kv_start = spans[first_visible][0]
            if groups and groups[-1][2] == kv_start and groups[-1][1] == suffix_start:
                previous_q_start, _previous_q_end, previous_kv_start, _previous_kv_end = groups[-1]
                groups[-1] = (previous_q_start, seq_len, previous_kv_start, seq_len)
            else:
                groups.append((suffix_start, seq_len, kv_start, seq_len))
        return groups

    def _scalar_rope_position(self, value: int, device: torch.device) -> torch.Tensor:
        return torch.full((3, 1), fill_value=int(value), device=device, dtype=torch.long)

    def _attention_module(self, layer):
        for name in ("self_attn", "attention", "attn"):
            attn = getattr(layer, name, None)
            if attn is not None:
                return attn
        raise RuntimeError(f"Could not locate self attention module in layer {type(layer)!r}.")

    def _repeat_kv(self, states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep <= 1:
            return states
        bsz, num_kv_heads, seq_len, head_dim = states.shape
        states = states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
        return states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)

    def _uses_flash_attention_2(self) -> bool:
        return _get_active_attn_implementation(self.model) == "flash_attention_2"

    def _flash_attend_strict(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        *,
        causal: bool = True,
        window_left: Optional[int] = None,
    ) -> torch.Tensor:
        if q_states.device.type != "cuda":
            raise RuntimeError("ShallowStream_Qwen3VL_V3 lower attention requires CUDA for flash-attn.")
        if q_states.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"ShallowStream_Qwen3VL_V3 lower attention requires fp16/bf16 for flash-attn, got {q_states.dtype}."
            )
        try:
            from flash_attn import flash_attn_func  # type: ignore
        except Exception as exc:
            raise RuntimeError("ShallowStream_Qwen3VL_V3 requires flash-attn when using flash_attention_2.") from exc

        q_bshd = q_states.transpose(1, 2).contiguous()
        k_bshd = k_states.transpose(1, 2).contiguous()
        v_bshd = v_states.transpose(1, 2).contiguous()
        kwargs: Dict[str, Any] = {
            "dropout_p": 0.0,
            "softmax_scale": None,
            "causal": bool(causal),
        }
        if window_left is not None:
            kwargs["window_size"] = (max(0, int(window_left)), 0)
        result = flash_attn_func(q_bshd, k_bshd, v_bshd, **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        return result.transpose(1, 2).contiguous()
    def _rekv_sink_enabled(self) -> bool:
        enabled = _as_bool(self.config, "use_rekv_sink")
        if enabled:
            self._local_window_frames()
        return enabled

    def _bump_rekv_sink_path(self, name: str) -> None:
        stats = getattr(self, "_rekv_sink_path_stats", None)
        if not isinstance(stats, dict):
            stats = {}
            self._rekv_sink_path_stats = stats
        stats[name] = int(stats.get(name, 0)) + 1

    def _attend_with_physical_sink_local(
        self,
        q_raw: torch.Tensor,
        main_k_raw: torch.Tensor,
        main_v: torch.Tensor,
        q_positions: torch.Tensor,
        main_positions: torch.Tensor,
        rotary_emb,
        *,
        past_len: int,
        sink_len: int = 0,
        sink_entry: Optional[Dict[str, torch.Tensor]] = None,
        local_window_tokens: int,
        attention_unit_groups: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> torch.Tensor:
        """Run maskless causal FlashAttention over physically retained KV.

        The persistent sink is prepended to an exact local KV slice. Because
        current queries are the suffix of that concatenation, FlashAttention's
        bottom-right causal alignment exposes every sink/local past token and
        only the causal prefix of the current unit.
        """

        if not self._uses_flash_attention_2():
            raise RuntimeError("physical sink/local attention requires flash_attention_2")
        q_len = int(q_raw.shape[2])
        main_len = int(main_k_raw.shape[2])
        n_local = int(local_window_tokens)
        if n_local <= 0:
            raise ValueError("local_window_tokens must be positive")

        external_sink_k = None
        external_sink_v = None
        external_sink_positions = None
        if isinstance(sink_entry, dict) and isinstance(sink_entry.get("k"), torch.Tensor):
            external_sink_k = sink_entry["k"].to(device=main_k_raw.device, dtype=main_k_raw.dtype)
            external_sink_v = sink_entry["v"].to(device=main_v.device, dtype=main_v.dtype)
            external_sink_positions = sink_entry.get("positions")
            if not isinstance(external_sink_positions, torch.Tensor):
                raise RuntimeError("stream sink KV is missing positions")
            external_sink_positions = external_sink_positions.to(device=main_positions.device, dtype=torch.long)
            external_sink_len = int(external_sink_k.shape[2])
            if (
                int(external_sink_v.shape[2]) != external_sink_len
                or int(external_sink_positions.shape[-1]) != external_sink_len
            ):
                raise RuntimeError("stream sink K/V/position lengths do not match")
        else:
            external_sink_len = 0

        groups: List[Tuple[int, int, int, int]] = []
        if attention_unit_groups is not None:
            expected_q_start = 0
            for group in attention_unit_groups:
                if len(group) != 4:
                    raise RuntimeError(f"invalid attention unit group: {group!r}")
                q_start, q_end, kv_start, kv_end = (int(value) for value in group)
                if q_start != expected_q_start or not (q_start < q_end <= q_len):
                    raise RuntimeError(f"attention unit groups do not contiguously cover queries: {group!r}")
                if not (0 <= kv_start < kv_end <= main_len):
                    raise RuntimeError(f"attention unit group has invalid KV bounds: {group!r}")
                if kv_end != int(past_len) + q_end:
                    raise RuntimeError(
                        f"attention unit group is not bottom-right causal aligned: {group!r}, "
                        f"past_len={past_len}"
                    )
                if kv_end - kv_start < q_end - q_start:
                    raise RuntimeError(f"attention unit group has fewer keys than queries: {group!r}")
                groups.append((q_start, q_end, kv_start, kv_end))
                expected_q_start = q_end
            if expected_q_start != q_len:
                raise RuntimeError(
                    f"attention unit groups cover {expected_q_start} queries, expected {q_len}"
                )
        else:
            local_keep = max(n_local, q_len)
            groups = [(0, q_len, max(0, main_len - local_keep), main_len)]

        q_rot = self._apply_rope_to_key(q_raw, q_positions, rotary_emb)
        outputs: List[torch.Tensor] = []
        for q_start, q_end, kv_start, kv_end in groups:
            local_start = kv_start
            prefix_k: List[torch.Tensor] = []
            prefix_v: List[torch.Tensor] = []
            prefix_positions: List[torch.Tensor] = []
            if external_sink_len > 0:
                assert external_sink_k is not None
                assert external_sink_v is not None
                assert external_sink_positions is not None
                prefix_k.append(external_sink_k)
                prefix_v.append(external_sink_v)
                prefix_positions.append(external_sink_positions)
            else:
                internal_sink_len = max(0, min(int(sink_len), main_len))
                if internal_sink_len > 0 and local_start >= internal_sink_len:
                    prefix_k.append(main_k_raw[:, :, :internal_sink_len, :])
                    prefix_v.append(main_v[:, :, :internal_sink_len, :])
                    prefix_positions.append(main_positions[..., :internal_sink_len])
                elif internal_sink_len > 0:
                    local_start = 0

            prefix_k.append(main_k_raw[:, :, local_start:kv_end, :])
            prefix_v.append(main_v[:, :, local_start:kv_end, :])
            prefix_positions.append(main_positions[..., local_start:kv_end])
            physical_k_raw = torch.cat(prefix_k, dim=2)
            physical_v = torch.cat(prefix_v, dim=2)
            physical_positions = torch.cat(prefix_positions, dim=-1)
            physical_k = self._apply_rope_to_key(physical_k_raw, physical_positions, rotary_emb)
            group_q = q_rot[:, :, q_start:q_end, :]
            if int(physical_k.shape[2]) < int(group_q.shape[2]):
                raise RuntimeError("physical sink/local KV has fewer keys than current queries")
            outputs.append(
                self._flash_attend_strict(
                    group_q,
                    physical_k,
                    physical_v,
                    causal=True,
                )
            )

        self._bump_rekv_sink_path(
            "flash_physical_unit_sink_local"
            if attention_unit_groups is not None
            else "flash_physical_sink_local"
        )
        return torch.cat(outputs, dim=2)

    def _attend_with_rekv_sink(
        self,
        q_raw: torch.Tensor,
        main_k_raw: torch.Tensor,
        main_v: torch.Tensor,
        q_positions: torch.Tensor,
        main_positions: torch.Tensor,
        rotary_emb,
        *,
        past_len: int,
        sink_len: int,
        sink_entry: Optional[Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        local_window_tokens: int,
        attention_unit_groups: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> torch.Tensor:
        """Attend once over a physically concatenated sink prefix and local suffix."""

        if attention_mask is not None:
            raise RuntimeError(
                "ReKV sink attention no longer supports dense per-query masks; "
                "provide physical attention_unit_groups instead"
            )
        if not self._uses_flash_attention_2():
            raise RuntimeError("ReKV sink attention requires flash_attention_2")
        return self._attend_with_physical_sink_local(
            q_raw,
            main_k_raw,
            main_v,
            q_positions,
            main_positions,
            rotary_emb,
            past_len=past_len,
            sink_len=sink_len,
            sink_entry=sink_entry,
            local_window_tokens=int(local_window_tokens),
            attention_unit_groups=attention_unit_groups,
        )
    def _qwen3_qkv_raw(self, attn, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        head_dim = int(getattr(attn, "head_dim"))
        q = attn.q_proj(hidden_states).view(bsz, seq_len, -1, head_dim)
        k = attn.k_proj(hidden_states).view(bsz, seq_len, -1, head_dim)
        v = attn.v_proj(hidden_states).view(bsz, seq_len, -1, head_dim)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        return (
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )

    def _forward_lower_layers_raw(
        self,
        hidden_states: torch.Tensor,
        layers,
        rotary_emb,
        start_layer: int,
        end_layer: int,
        past_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]],
        positions: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        update_cache: bool = True,
        capture_q_layers: Optional[Sequence[int]] = None,
        capture_hidden_layers: Optional[Sequence[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_unit_groups: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        cache_policy: Optional[str] = None,
        sink_len: int = 0,
        sink_raw_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        local_window_tokens: Optional[int] = None,
        consume_past_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        new_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        capture_q_set = {int(layer_idx) for layer_idx in (capture_q_layers or [])}
        capture_hidden_set = {int(layer_idx) for layer_idx in (capture_hidden_layers or [])}
        resolved_cache_policy = cache_policy or ("append" if update_cache else "reuse")
        if resolved_cache_policy not in {"append", "capture_current", "reuse"}:
            raise ValueError(f"Unsupported lower cache_policy={resolved_cache_policy!r}")

        for layer_idx in range(start_layer, end_layer):
            layer = layers[layer_idx]
            layer_device = self._layer_device(layer, hidden_states.device)
            hidden_states = hidden_states.to(layer_device)
            layer_positions = positions.to(device=layer_device, dtype=torch.long)
            if int(layer_positions.shape[-1]) != int(hidden_states.shape[1]):
                raise RuntimeError(
                    f"position count {int(layer_positions.shape[-1])} != hidden length {int(hidden_states.shape[1])}."
                )

            attn = self._attention_module(layer)
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            q_raw, k_raw, v_raw = self._qwen3_qkv_raw(attn, normed)

            past_entry = past_raw_kv.get(layer_idx) if isinstance(past_raw_kv, dict) else None
            if isinstance(past_entry, dict) and isinstance(past_entry.get("k"), torch.Tensor):
                past_k = past_entry["k"].to(device=k_raw.device, dtype=k_raw.dtype)
                past_v = past_entry["v"].to(device=v_raw.device, dtype=v_raw.dtype)
                past_positions = past_entry.get("positions")
                if isinstance(past_positions, torch.Tensor):
                    past_positions = past_positions.to(device=layer_device, dtype=torch.long)
                else:
                    past_positions = self._scalar_rope_position(0, layer_device)[:, :0]
            else:
                past_k = k_raw[:, :, :0, :]
                past_v = v_raw[:, :, :0, :]
                past_positions = layer_positions[:, :0] if layer_positions.dim() == 2 else layer_positions[:0]

            total_k_raw = torch.cat([past_k, k_raw], dim=2)
            total_v = torch.cat([past_v, v_raw], dim=2)
            total_positions = torch.cat([past_positions, layer_positions], dim=-1)

            q_len = int(q_raw.shape[2])
            past_len = int(past_k.shape[2])
            layer_attention_mask = None
            if isinstance(attention_mask, torch.Tensor):
                if (
                    int(attention_mask.shape[-2]) != q_len
                    or int(attention_mask.shape[-1]) != int(total_k_raw.shape[2])
                ):
                    raise RuntimeError(
                        f"attention mask shape {tuple(attention_mask.shape)} does not match "
                        f"q_len={q_len}, kv_len={int(total_k_raw.shape[2])}."
                    )
                layer_attention_mask = attention_mask.to(device=q_raw.device, dtype=q_raw.dtype)

            if self._rekv_sink_enabled():
                if local_window_tokens is None:
                    raise ValueError("local_window_tokens is required when use_rekv_sink=true")
                sink_entry = sink_raw_kv.get(layer_idx) if isinstance(sink_raw_kv, dict) else None
                attn_output = self._attend_with_rekv_sink(
                    q_raw,
                    total_k_raw,
                    total_v,
                    layer_positions,
                    total_positions,
                    rotary_emb,
                    past_len=past_len,
                    sink_len=sink_len,
                    sink_entry=sink_entry,
                    attention_mask=layer_attention_mask,
                    attention_unit_groups=attention_unit_groups,
                    local_window_tokens=int(local_window_tokens),
                )
            else:
                if (
                    attention_unit_groups is not None
                    and layer_attention_mask is None
                    and self._uses_flash_attention_2()
                ):
                    if local_window_tokens is None:
                        raise ValueError("local_window_tokens is required for physical unit attention")
                    attn_output = self._attend_with_physical_sink_local(
                        q_raw,
                        total_k_raw,
                        total_v,
                        layer_positions,
                        total_positions,
                        rotary_emb,
                        past_len=past_len,
                        local_window_tokens=int(local_window_tokens),
                        attention_unit_groups=attention_unit_groups,
                    )
                else:
                    q_rot = self._apply_rope_to_key(q_raw, layer_positions, rotary_emb)
                    k_rot = self._apply_rope_to_key(total_k_raw, total_positions, rotary_emb)
                if (
                    attention_unit_groups is None
                    and layer_attention_mask is None
                    and self._uses_flash_attention_2()
                ):
                    attn_output = self._flash_attend_strict(q_rot, k_rot, total_v, causal=True)
                elif attention_unit_groups is None or layer_attention_mask is not None:
                    v_use = total_v
                    if int(k_rot.shape[1]) != int(q_rot.shape[1]):
                        if int(q_rot.shape[1]) % int(k_rot.shape[1]) != 0:
                            raise RuntimeError(
                                f"Q/KV head mismatch at layer {layer_idx}: q={q_rot.shape[1]} kv={k_rot.shape[1]}"
                            )
                        n_rep = int(q_rot.shape[1]) // int(k_rot.shape[1])
                        k_rot = self._repeat_kv(k_rot, n_rep)
                        v_use = self._repeat_kv(v_use, n_rep)
                    use_sdpa_causal = layer_attention_mask is None and past_len == 0 and q_len > 1
                    sdpa_mask = layer_attention_mask
                    if sdpa_mask is None and not use_sdpa_causal:
                        sdpa_mask = self._causal_mask(normed, q_len, past_len)
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q_rot,
                        k_rot,
                        v_use,
                        attn_mask=sdpa_mask,
                        dropout_p=0.0,
                        is_causal=use_sdpa_causal,
                    )
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                int(normed.shape[0]),
                int(normed.shape[1]),
                -1,
            )
            hidden_states = residual + attn.o_proj(attn_output)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states=hidden_states,
                    visual_pos_masks=visual_pos_masks,
                    visual_embeds=deepstack_visual_embeds[layer_idx],
                )

            if resolved_cache_policy == "append":
                cache_entry = {
                    "k": total_k_raw.detach(),
                    "v": total_v.detach(),
                    "positions": total_positions.detach(),
                }
                if layer_idx in capture_q_set:
                    cache_entry["q"] = q_raw.detach()
                if layer_idx in capture_hidden_set:
                    cache_entry["hidden"] = hidden_states.detach()
                new_cache[layer_idx] = cache_entry
            elif resolved_cache_policy == "capture_current":
                capture_device = (
                    self._streaming_archive_device()
                    if self._streaming_archive_enabled()
                    else q_raw.device
                )
                cache_entry = {
                    "k": k_raw.detach().to(capture_device),
                    "v": v_raw.detach().to(capture_device),
                    "positions": layer_positions.detach().to(capture_device),
                }
                if layer_idx in capture_q_set:
                    cache_entry["q"] = q_raw.detach().to(capture_device)
                if layer_idx in capture_hidden_set:
                    cache_entry["hidden"] = hidden_states.detach().to(capture_device)
                new_cache[layer_idx] = cache_entry
            else:
                new_cache[layer_idx] = {
                    "k": past_k.detach(),
                    "v": past_v.detach(),
                    "positions": past_positions.detach(),
                }

            if consume_past_cache and isinstance(past_raw_kv, dict):
                # The returned cache supersedes this layer's input cache. Pop it
                # immediately so long FullKV contexts never coexist as two
                # complete dictionaries while preserving identical concatenated
                # K/V tensors and positions.
                past_raw_kv.pop(layer_idx, None)

        return hidden_states, new_cache

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if visual_pos_masks is None:
            return hidden_states
        mask = visual_pos_masks.to(hidden_states.device)
        embeds = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        expected = int(mask.sum().item())
        if expected != int(embeds.shape[0]):
            raise RuntimeError(
                f"Qwen3 DeepStack visual token mismatch: mask selects {expected}, "
                f"but embeddings contain {int(embeds.shape[0])}."
            )
        hidden_states = hidden_states.clone()
        hidden_states[mask, :] = hidden_states[mask, :] + embeds
        return hidden_states

    def _index_cache_positions(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device=tensor.device)
        return tensor.index_select(2, positions)

    def _index_hidden_positions(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device=hidden.device)
        return hidden.index_select(1, positions)

    def _normalized_key_vector(self, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        selected = self._index_cache_positions(key, positions).float()
        vec = selected.mean(dim=2).reshape(-1)
        return torch.nn.functional.normalize(vec, dim=0)

    def _normalized_key_vectors(self, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        selected = self._index_cache_positions(key, positions).float()
        vectors = selected.permute(2, 0, 1, 3).contiguous().view(int(selected.shape[2]), -1)
        return torch.nn.functional.normalize(vectors, dim=1)

    def _normalized_query_vector(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        key_head_count: int,
    ) -> torch.Tensor:
        selected = self._index_cache_positions(query, positions).float()
        if int(selected.shape[1]) != int(key_head_count):
            if int(selected.shape[1]) % int(key_head_count) != 0:
                raise RuntimeError(
                    f"Q/KV head mismatch for retrieval query: q_heads={int(selected.shape[1])}, "
                    f"kv_heads={int(key_head_count)}"
                )
            group = int(selected.shape[1]) // int(key_head_count)
            selected = selected.view(
                int(selected.shape[0]),
                int(key_head_count),
                group,
                int(selected.shape[2]),
                int(selected.shape[3]),
            ).mean(dim=2)
        vec = selected.mean(dim=2).reshape(-1)
        return torch.nn.functional.normalize(vec, dim=0)
