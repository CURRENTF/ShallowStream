"""Typed KV, cluster, and stream-session state for ShallowStream Qwen3-VL V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.shallowstream.common import LayerIndexedLegacyCache


@dataclass
class _FrameKVState:
    frame_id: int
    sample_index: int
    timestamp: float
    token_indices: torch.Tensor
    # Visual placeholder tokens only live in token_indices. context_indices also
    # includes text tokens between the previous video unit and this unit, which
    # is where Qwen3-VL places explicit timestamp tokens such as "<3.0 seconds>".
    context_indices: torch.Tensor
    context_positions: torch.Tensor
    positions: torch.Tensor
    key_vec: torch.Tensor
    visual_embeds: Optional[torch.Tensor] = None
    deepstack_embeds: Optional[List[torch.Tensor]] = None
    grid_thw: Optional[torch.Tensor] = None
    image: Optional[Image.Image] = None
    source_images: Optional[List[Image.Image]] = None
    source_sample_indices: Optional[List[int]] = None
    source_timestamps: Optional[List[float]] = None


@dataclass
class _RecentSourceFrame:
    """One distinct sampled RGB frame retained for the final recent-image prompt."""

    sample_index: int
    timestamp: float
    image: Image.Image


@dataclass
class _LongKVCluster:
    start_frame: int
    end_frame: int
    count: int
    positions: torch.Tensor
    key_vec: torch.Tensor
    lower_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    hidden: torch.Tensor
    start_time: float
    end_time: float
    deepstack_embeds: Optional[List[torch.Tensor]] = None
    visual_embeds: Optional[torch.Tensor] = None
    grid_thw: Optional[torch.Tensor] = None
    representative_image: Optional[Image.Image] = None
    representative_key_vec: Optional[torch.Tensor] = None

    @classmethod
    def from_frame(
        cls,
        frame: _FrameKVState,
        lower_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        hidden: torch.Tensor,
        deepstack_embeds: Optional[List[torch.Tensor]],
    ) -> "_LongKVCluster":
        return cls(
            start_frame=frame.frame_id,
            end_frame=frame.frame_id,
            count=1,
            positions=frame.positions.float().clone(),
            key_vec=frame.key_vec.clone(),
            lower_kv={idx: (kv[0].detach(), kv[1].detach()) for idx, kv in lower_kv.items()},
            hidden=hidden.detach().clone(),
            start_time=frame.timestamp,
            end_time=frame.timestamp,
            deepstack_embeds=(
                [emb.detach().clone() for emb in deepstack_embeds]
                if deepstack_embeds is not None
                else None
            ),
            visual_embeds=(
                frame.visual_embeds.detach().clone()
                if isinstance(frame.visual_embeds, torch.Tensor)
                else None
            ),
            grid_thw=(
                frame.grid_thw.detach().clone()
                if isinstance(frame.grid_thw, torch.Tensor)
                else None
            ),
            representative_image=(
                frame.image.copy()
                if isinstance(frame.image, Image.Image)
                else None
            ),
            representative_key_vec=(
                frame.key_vec.detach().clone()
                if isinstance(frame.image, Image.Image)
                else None
            ),
        )

    def merge(
        self,
        frame: _FrameKVState,
        frame_lower_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        frame_hidden: torch.Tensor,
        frame_deepstack_embeds: Optional[List[torch.Tensor]],
    ) -> None:
        new_count = self.count + 1
        self.positions = (self.positions.float() * self.count + frame.positions.float()) / new_count
        self.key_vec = torch.nn.functional.normalize(
            (self.key_vec.float() * self.count + frame.key_vec.float()) / new_count,
            dim=0,
        ).to(self.key_vec.dtype)
        if isinstance(frame.image, Image.Image):
            representative_key = self.representative_key_vec
            if (
                not isinstance(self.representative_image, Image.Image)
                or not isinstance(representative_key, torch.Tensor)
            ):
                self.representative_image = frame.image.copy()
                self.representative_key_vec = frame.key_vec.detach().clone()
            elif representative_key.numel() == self.key_vec.numel():
                old_similarity = torch.dot(
                    representative_key.float(),
                    self.key_vec.float(),
                ).item()
                new_similarity = torch.dot(
                    frame.key_vec.float(),
                    self.key_vec.float(),
                ).item()
                if new_similarity > old_similarity:
                    self.representative_image = frame.image.copy()
                    self.representative_key_vec = frame.key_vec.detach().clone()
        for layer_idx, (old_k, old_v) in self.lower_kv.items():
            new_k, new_v = frame_lower_kv[layer_idx]
            new_k = new_k.to(device=old_k.device, dtype=old_k.dtype)
            new_v = new_v.to(device=old_v.device, dtype=old_v.dtype)
            self.lower_kv[layer_idx] = (
                (old_k * self.count + new_k) / new_count,
                (old_v * self.count + new_v) / new_count,
            )
        frame_hidden = frame_hidden.to(device=self.hidden.device, dtype=self.hidden.dtype)
        self.hidden = (self.hidden * self.count + frame_hidden) / new_count
        if self.deepstack_embeds is not None and frame_deepstack_embeds is not None:
            if len(self.deepstack_embeds) == len(frame_deepstack_embeds):
                merged_deepstack: List[torch.Tensor] = []
                for old_emb, new_emb in zip(self.deepstack_embeds, frame_deepstack_embeds):
                    new_emb = new_emb.to(device=old_emb.device, dtype=old_emb.dtype)
                    merged_deepstack.append((old_emb * self.count + new_emb) / new_count)
                self.deepstack_embeds = merged_deepstack
            else:
                self.deepstack_embeds = None
        elif self.deepstack_embeds is not None or frame_deepstack_embeds is not None:
            self.deepstack_embeds = None
        if self.visual_embeds is not None and isinstance(frame.visual_embeds, torch.Tensor):
            frame_visual = frame.visual_embeds.to(device=self.visual_embeds.device, dtype=self.visual_embeds.dtype)
            if tuple(frame_visual.shape) == tuple(self.visual_embeds.shape):
                self.visual_embeds = (self.visual_embeds * self.count + frame_visual) / new_count
            else:
                self.visual_embeds = None
                self.grid_thw = None
        elif self.visual_embeds is not None or isinstance(frame.visual_embeds, torch.Tensor):
            self.visual_embeds = None
            self.grid_thw = None
        self.count = new_count
        self.end_frame = frame.frame_id
        self.end_time = frame.timestamp

@dataclass
class _Qwen3VLStreamSession:
    video_key: str
    # The first batch's text prefix is shallow-prefilled once and reused when
    # selected visual context is compacted for internal-KV continuation.
    prompt_prefix_lower_kv: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    prompt_prefix_hidden_after_prune: Optional[torch.Tensor] = None
    prompt_prefix_len: int = 0
    # Initial text-prefix KV is kept separately from visual history so frame
    # indices remain visual-cache-relative while every lower-layer stage can
    # still attend to the same persistent ReKV sink.
    sink_lower_kv: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    sink_len: int = 0
    local_window_tokens: int = 0
    # Retrieval/detail raw pre-RoPE history. Archive mode keeps it exact on the
    # configured archive device; normal streaming may compact promoted units.
    raw_lower_kv: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    # Bounded suffix used only for the next streaming video append.
    active_lower_kv: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    hidden_after_prune: Optional[torch.Tensor] = None
    frame_states: List[_FrameKVState] = field(default_factory=list)
    # Keep recent RGB frames independently from Qwen temporal video units. The
    # video path may duplicate its final frame to satisfy temporal patching;
    # the answer prompt must still receive the latest distinct sampled images.
    recent_source_frames: List[_RecentSourceFrame] = field(default_factory=list)
    clusters: List[_LongKVCluster] = field(default_factory=list)
    token_frame_ids: Optional[torch.Tensor] = None
    visual_pos_masks: Optional[torch.Tensor] = None
    deepstack_visual_embeds: Optional[List[torch.Tensor]] = None
    last_timestamp: float = -1.0
    next_frame_id: int = 0
    next_position: int = 0
    visual_rope_base: Optional[torch.Tensor] = None
    next_visual_temporal_position: int = 0


class _SelectedCache(LayerIndexedLegacyCache):
    """Compatibility name for the shared layer-indexed HF cache interface."""


class _StopAfterInputEmbeds(RuntimeError):
    pass
