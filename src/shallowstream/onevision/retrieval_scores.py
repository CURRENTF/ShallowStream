"""Visual/subtitle score fusion for OneVision frame retrieval."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class OneVisionRetrievalScoreMixin:
    def _get_retrieval_weights(self) -> Tuple[float, float]:
        """Return non-negative retrieval fusion weights (visual, subtitle)."""
        v_w = max(0.0, float(self.config.get("retrieval_visual_weight", 0.7)))
        s_w = max(0.0, float(self.config.get("retrieval_subtitle_weight", 0.3)))
        # Keep visual retrieval as safe fallback if both are disabled by config.
        if v_w <= 0.0 and s_w <= 0.0:
            v_w = 1.0
        return v_w, s_w

    def _get_frame_caption_available(self, num_frames: int) -> List[bool]:
        """Cheap per-frame subtitle availability without touching attention tensors."""
        frame_spans = self.state.get("frame_spans")
        if not isinstance(frame_spans, list) or num_frames <= 0:
            return [False] * max(0, int(num_frames))
        out: List[bool] = []
        for i in range(num_frames):
            if i >= len(frame_spans):
                out.append(False)
                continue
            span = frame_spans[i]
            st = int(span.get("caption_start", -1))
            ed = int(span.get("caption_end", -1))
            out.append(ed > st and st >= 0)
        return out

    def _fuse_retrieval_scores(
        self,
        visual_scores: torch.Tensor,
        subtitle_scores: Optional[torch.Tensor],
        caption_mask: Optional[List[bool]],
    ) -> torch.Tensor:
        v_w, s_w = self._get_retrieval_weights()
        if subtitle_scores is None or subtitle_scores.numel() == 0 or not isinstance(caption_mask, list):
            return visual_scores
        if subtitle_scores.shape != visual_scores.shape:
            return visual_scores

        mask = torch.tensor(caption_mask, device=visual_scores.device, dtype=torch.bool)
        if mask.numel() != visual_scores.numel():
            return visual_scores
        mask_f = mask.to(dtype=visual_scores.dtype)

        if s_w <= 0.0:
            return visual_scores
        if v_w <= 0.0:
            return torch.where(mask, subtitle_scores, visual_scores)

        num = (v_w * visual_scores) + (s_w * mask_f * subtitle_scores)
        den = v_w + (s_w * mask_f)
        return num / torch.clamp(den, min=1e-8)

    def _observe_retrieval(
        self,
        *,
        scores: torch.Tensor,
        candidate_idx_t: torch.Tensor,
        recent_n: int,
        score_order: str,
        video_path: Optional[str],
    ) -> None:
        """Hook for observation-only variants; selection must remain unchanged."""
