from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from transformers import (
    AutoConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from src.config import apply_config_sources
from src.shallowstream.common import (
    SingleLayerLegacyCache,
    expand_temporal_neighbors,
    get_active_attn_implementation,
)
from src.utils.eval_io import atomic_write_json
from src.utils.time_trace import (
    StageRecorder,
    extract_clip_range_seconds,
    extract_sample_id,
    make_prompt_preview,
)

from .config import (
    LONG_CLUSTER_COSINE_SIM_THRESHOLD,
    NO_NEW_VIDEO_CHUNK,
    ONEVISION_V3_DEFAULT_CONFIG,
    new_runtime_state,
)


class OneVisionMemoryMixin:
    def _short_window_start(self, total_frames: int) -> int:
        search_last_n_cfg = max(0, int(self.config.get("retrieval_search_last_n_frames", 0)))
        if search_last_n_cfg <= 0:
            return 0
        return max(0, int(total_frames) - search_last_n_cfg)

    def _extract_frame_visual_kv_per_layer(self, frame_idx: int, prune: int) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
        lower_kv = self.state.get("lower_kv")
        frame_spans = self.state.get("frame_spans")
        if not isinstance(lower_kv, dict) or not isinstance(frame_spans, list):
            return None
        if frame_idx < 0 or frame_idx >= len(frame_spans):
            return None
        span = frame_spans[frame_idx]
        st = int(span.get("visual_start", -1))
        ed = int(span.get("visual_end", -1))
        tpf = int(self.config.get("n_frame_tokens", 196))
        if st < 0 or ed <= st or (ed - st) != tpf:
            return None

        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for lid in range(prune):
            entry = lower_kv.get(lid)
            if not isinstance(entry, dict):
                return None
            k = entry.get("k")
            v = entry.get("v")
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                return None
            seq_len = int(k.shape[-2])
            if ed > seq_len:
                return None
            out[lid] = {
                "k": k[:, :, st:ed, :].detach(),
                "v": v[:, :, st:ed, :].detach(),
            }
        return out

    def _frame_key_vec_from_visual_k(self, layer_idx: int, k_frame: torch.Tensor, device: str) -> torch.Tensor:
        layers, _, _, _ = self._get_lm_components()
        if layer_idx < 0 or layer_idx >= len(layers):
            return torch.empty((0,), dtype=torch.float16, device=device)
        attn = layers[layer_idx].self_attn
        num_heads, num_kv_heads, _head_dim = self._attention_shape(attn)
        tpf = int(self.config.get("n_frame_tokens", 196))
        if int(k_frame.shape[-2]) != tpf:
            return torch.empty((0,), dtype=torch.float16, device=device)

        k_use = k_frame
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            k_use = self._repeat_kv(k_use, n_rep)
        frame_vec = k_use[0].transpose(0, 1).reshape(tpf, -1).mean(dim=0)
        return torch.nn.functional.normalize(frame_vec.to(device=device), dim=-1)

    def _new_long_cluster(
        self,
        frame_hidden_l8: torch.Tensor,
        frame_visual_kv: Dict[int, Dict[str, torch.Tensor]],
        frame_key_vec: Optional[torch.Tensor] = None,
        frame_input_embed: Optional[torch.Tensor] = None,
        frame_source_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        cluster_kv: Dict[int, Dict[str, torch.Tensor]] = {}
        for lid, kv in frame_visual_kv.items():
            cluster_kv[int(lid)] = {
                "k": kv["k"].detach().clone(),
                "v": kv["v"].detach().clone(),
            }
        return {
            "count": 1,
            "hidden_l8": frame_hidden_l8.detach().clone(),
            "input_embeds": (
                frame_input_embed.detach().clone()
                if isinstance(frame_input_embed, torch.Tensor)
                else None
            ),
            "lower_kv": cluster_kv,
            "center_key_vec": (
                torch.nn.functional.normalize(
                    frame_key_vec.detach().clone().to(dtype=torch.float32),
                    dim=-1,
                ).to(dtype=frame_key_vec.dtype)
                if isinstance(frame_key_vec, torch.Tensor) and frame_key_vec.numel() > 0
                else None
            ),
            "source_image": (
                frame_source_image.copy()
                if isinstance(frame_source_image, Image.Image)
                else None
            ),
            "source_image_key_vec": (
                frame_key_vec.detach().clone()
                if isinstance(frame_source_image, Image.Image)
                and isinstance(frame_key_vec, torch.Tensor)
                else None
            ),
        }

    def _cluster_center_key_vec(
        self,
        cluster: Dict[str, Any],
        layer_idx: int,
        device: str,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        center_key_vec = cluster.get("center_key_vec")
        if isinstance(center_key_vec, torch.Tensor) and center_key_vec.numel() > 0:
            c_vec = torch.nn.functional.normalize(center_key_vec.to(device=device, dtype=torch.float32), dim=-1)
            cluster["center_key_vec"] = c_vec.detach().to(dtype=center_key_vec.dtype)
            return c_vec.to(dtype=out_dtype)

        c_lkv = cluster.get("lower_kv")
        if not isinstance(c_lkv, dict):
            return torch.empty((0,), dtype=out_dtype, device=device)
        c_entry = c_lkv.get(int(layer_idx))
        if not isinstance(c_entry, dict):
            return torch.empty((0,), dtype=out_dtype, device=device)
        ck = c_entry.get("k")
        if not isinstance(ck, torch.Tensor):
            return torch.empty((0,), dtype=out_dtype, device=device)
        c_vec = self._frame_key_vec_from_visual_k(layer_idx, ck.to(device=device), device=device)
        if c_vec.numel() == 0:
            return c_vec
        c_vec_norm = torch.nn.functional.normalize(c_vec.to(dtype=torch.float32), dim=-1)
        cluster["center_key_vec"] = c_vec_norm.detach().to(dtype=c_vec.dtype)
        return c_vec_norm.to(dtype=out_dtype)

    def _update_long_cluster_running_mean(
        self,
        cluster: Dict[str, Any],
        frame_hidden_l8: torch.Tensor,
        frame_visual_kv: Dict[int, Dict[str, torch.Tensor]],
        frame_key_vec: torch.Tensor,
        frame_input_embed: Optional[torch.Tensor] = None,
        frame_source_image: Optional[Image.Image] = None,
    ) -> None:
        cnt = max(1, int(cluster.get("count", 1)))
        den = float(cnt + 1)

        old_hidden = cluster.get("hidden_l8")
        if isinstance(old_hidden, torch.Tensor) and old_hidden.shape == frame_hidden_l8.shape:
            mixed_hidden = (old_hidden.to(dtype=torch.float32) * float(cnt) + frame_hidden_l8.to(dtype=torch.float32)) / den
            cluster["hidden_l8"] = mixed_hidden.to(dtype=frame_hidden_l8.dtype)
        else:
            cluster["hidden_l8"] = frame_hidden_l8.detach().clone()

        if isinstance(frame_input_embed, torch.Tensor):
            old_input = cluster.get("input_embeds")
            if isinstance(old_input, torch.Tensor) and old_input.shape == frame_input_embed.shape:
                mixed_input = (
                    old_input.to(dtype=torch.float32) * float(cnt)
                    + frame_input_embed.to(dtype=torch.float32)
                ) / den
                cluster["input_embeds"] = mixed_input.to(dtype=frame_input_embed.dtype)
            else:
                cluster["input_embeds"] = frame_input_embed.detach().clone()

        cluster_lkv = cluster.get("lower_kv")
        if not isinstance(cluster_lkv, dict):
            cluster_lkv = {}
            cluster["lower_kv"] = cluster_lkv
        for lid, kv in frame_visual_kv.items():
            old_entry = cluster_lkv.get(int(lid))
            if (
                isinstance(old_entry, dict)
                and isinstance(old_entry.get("k"), torch.Tensor)
                and isinstance(old_entry.get("v"), torch.Tensor)
                and old_entry["k"].shape == kv["k"].shape
                and old_entry["v"].shape == kv["v"].shape
            ):
                mixed_k = (old_entry["k"].to(dtype=torch.float32) * float(cnt) + kv["k"].to(dtype=torch.float32)) / den
                mixed_v = (old_entry["v"].to(dtype=torch.float32) * float(cnt) + kv["v"].to(dtype=torch.float32)) / den
                cluster_lkv[int(lid)] = {
                    "k": mixed_k.to(dtype=kv["k"].dtype),
                    "v": mixed_v.to(dtype=kv["v"].dtype),
                }
            else:
                cluster_lkv[int(lid)] = {
                    "k": kv["k"].detach().clone(),
                    "v": kv["v"].detach().clone(),
                }

        if isinstance(frame_key_vec, torch.Tensor) and frame_key_vec.numel() > 0:
            old_center = cluster.get("center_key_vec")
            if isinstance(old_center, torch.Tensor) and old_center.numel() == frame_key_vec.numel():
                mixed_center = (old_center.to(dtype=torch.float32) * float(cnt) + frame_key_vec.to(dtype=torch.float32)) / den
                cluster["center_key_vec"] = torch.nn.functional.normalize(mixed_center, dim=-1).to(dtype=frame_key_vec.dtype)
            else:
                cluster["center_key_vec"] = torch.nn.functional.normalize(
                    frame_key_vec.detach().clone().to(dtype=torch.float32),
                    dim=-1,
                ).to(dtype=frame_key_vec.dtype)

        if isinstance(frame_source_image, Image.Image):
            representative_key = cluster.get("source_image_key_vec")
            center_key = cluster.get("center_key_vec")
            if not isinstance(representative_key, torch.Tensor):
                cluster["source_image"] = frame_source_image.copy()
                cluster["source_image_key_vec"] = frame_key_vec.detach().clone()
            elif (
                isinstance(center_key, torch.Tensor)
                and representative_key.numel() == center_key.numel()
                and frame_key_vec.numel() == center_key.numel()
            ):
                center_float = center_key.to(
                    device=frame_key_vec.device,
                    dtype=torch.float32,
                )
                old_similarity = torch.dot(
                    representative_key.to(
                        device=frame_key_vec.device,
                        dtype=torch.float32,
                    ),
                    center_float,
                ).item()
                new_similarity = torch.dot(
                    frame_key_vec.to(dtype=torch.float32),
                    center_float,
                ).item()
                if new_similarity > old_similarity:
                    cluster["source_image"] = frame_source_image.copy()
                    cluster["source_image_key_vec"] = frame_key_vec.detach().clone()

        cluster["count"] = cnt + 1

    def _update_long_clusters_from_history(self, device: str) -> Dict[str, Any]:
        if bool(self.config.get("full_kv_mode", False)):
            return {"promoted_frames": 0, "long_cluster_count": 0}
        frame_hidden = self.state.get("frame_hidden_l8")
        frame_spans = self.state.get("frame_spans")
        lower_kv = self.state.get("lower_kv")
        source_ts = self.state.get("frame_source_ids")
        if (
            not isinstance(frame_hidden, torch.Tensor)
            or frame_hidden.numel() == 0
            or not isinstance(frame_spans, list)
            or not isinstance(lower_kv, dict)
        ):
            return {"promoted_frames": 0, "long_cluster_count": 0}

        total_frames = int(frame_hidden.shape[0])
        short_start = self._short_window_start(total_frames)
        if short_start <= 0:
            return {"promoted_frames": 0, "long_cluster_count": len(self.state.get("long_clusters") or [])}

        layers, _, _, _ = self._get_lm_components()
        prune = min(int(self.config.get("prune_layer", 8)), len(layers))
        if prune <= 0:
            return {"promoted_frames": 0, "long_cluster_count": len(self.state.get("long_clusters") or [])}

        clusters = self.state.get("long_clusters")
        if not isinstance(clusters, list):
            clusters = []
            self.state["long_clusters"] = clusters

        promoted = 0
        promote_n = int(short_start)
        frame_input_embeds = self.state.get("frame_input_embeds")
        frame_source_images = self.state.get("frame_source_images")
        if not isinstance(frame_source_images, list):
            frame_source_images = []
        for i in range(promote_n):
            visual_kv = self._extract_frame_visual_kv_per_layer(i, prune)
            if not isinstance(visual_kv, dict):
                break
            frame_key_vec = self._frame_key_vec_from_visual_k(prune - 1, visual_kv[prune - 1]["k"], device=device)
            if frame_key_vec.numel() == 0:
                break
            if i >= int(frame_hidden.shape[0]):
                break
            frame_hidden_l8 = frame_hidden[i].detach()
            frame_input_embed = (
                frame_input_embeds[i].detach()
                if isinstance(frame_input_embeds, torch.Tensor)
                and i < int(frame_input_embeds.shape[0])
                else None
            )
            frame_source_image = (
                frame_source_images[i]
                if i < len(frame_source_images)
                and isinstance(frame_source_images[i], Image.Image)
                else None
            )

            if len(clusters) == 0:
                clusters.append(
                    self._new_long_cluster(
                        frame_hidden_l8=frame_hidden_l8,
                        frame_visual_kv=visual_kv,
                        frame_key_vec=frame_key_vec,
                        frame_input_embed=frame_input_embed,
                        frame_source_image=frame_source_image,
                    )
                )
            else:
                last_cluster = clusters[-1]
                last_key = self._cluster_center_key_vec(
                    cluster=last_cluster,
                    layer_idx=prune - 1,
                    device=device,
                    out_dtype=frame_key_vec.dtype,
                )
                if last_key.numel() == 0 or last_key.numel() != frame_key_vec.numel():
                    clusters.append(
                        self._new_long_cluster(
                            frame_hidden_l8=frame_hidden_l8,
                            frame_visual_kv=visual_kv,
                            frame_key_vec=frame_key_vec,
                            frame_input_embed=frame_input_embed,
                            frame_source_image=frame_source_image,
                        )
                    )
                else:
                    sim = torch.nn.functional.cosine_similarity(
                        frame_key_vec.to(dtype=torch.float32).unsqueeze(0),
                        last_key.to(dtype=torch.float32).unsqueeze(0),
                        dim=-1,
                    ).item()
                    if float(sim) < self.long_cluster_cosine_sim_threshold:
                        clusters.append(
                            self._new_long_cluster(
                                frame_hidden_l8=frame_hidden_l8,
                                frame_visual_kv=visual_kv,
                                frame_key_vec=frame_key_vec,
                                frame_input_embed=frame_input_embed,
                                frame_source_image=frame_source_image,
                            )
                        )
                    else:
                        self._update_long_cluster_running_mean(
                            cluster=last_cluster,
                            frame_hidden_l8=frame_hidden_l8,
                            frame_visual_kv=visual_kv,
                            frame_key_vec=frame_key_vec,
                            frame_input_embed=frame_input_embed,
                            frame_source_image=frame_source_image,
                        )

            promoted += 1

        # Trim short-term buffers once after processing all promoted frames.
        if promoted > 0:
            if int(frame_hidden.shape[0]) > promoted:
                frame_hidden = frame_hidden[promoted:].contiguous()
            else:
                frame_hidden = frame_hidden[:0]

            if isinstance(frame_input_embeds, torch.Tensor) and frame_input_embeds.numel() > 0:
                if int(frame_input_embeds.shape[0]) > promoted:
                    frame_input_embeds = frame_input_embeds[promoted:].contiguous()
                else:
                    frame_input_embeds = frame_input_embeds[:0]

            if isinstance(frame_spans, list):
                frame_spans = frame_spans[promoted:]
            if isinstance(source_ts, list):
                source_ts = source_ts[promoted:]

            frame_captions = self.state.get("frame_captions")
            if isinstance(frame_captions, list):
                self.state["frame_captions"] = frame_captions[promoted:]

            frame_caption_hidden_l8 = self.state.get("frame_caption_hidden_l8")
            if isinstance(frame_caption_hidden_l8, list):
                self.state["frame_caption_hidden_l8"] = frame_caption_hidden_l8[promoted:]

            frame_debug_thumbs = self.state.get("frame_debug_thumbs")
            if isinstance(frame_debug_thumbs, list):
                self.state["frame_debug_thumbs"] = frame_debug_thumbs[promoted:]

            frame_evidence_images = self.state.get("frame_evidence_images")
            if isinstance(frame_evidence_images, list):
                self.state["frame_evidence_images"] = frame_evidence_images[promoted:]

            if isinstance(frame_source_images, list):
                frame_source_images = frame_source_images[promoted:]

        self.state["frame_hidden_l8"] = frame_hidden
        self.state["frame_input_embeds"] = frame_input_embeds
        self.state["frame_source_images"] = frame_source_images
        self.state["frame_spans"] = frame_spans
        self.state["frame_source_ids"] = source_ts
        return {"promoted_frames": int(promoted), "long_cluster_count": int(len(clusters))}

    def _select_long_clusters_by_query(
        self,
        layer_idx: int,
        q_vec: torch.Tensor,
        temp: float,
        score_order: str,
        device: str,
    ) -> Tuple[List[int], List[float]]:
        clusters = self.state.get("long_clusters")
        if not isinstance(clusters, list) or len(clusters) == 0:
            return [], []

        vecs: List[torch.Tensor] = []
        raw_idx: List[int] = []
        for i, cluster in enumerate(clusters):
            c_vec = self._cluster_center_key_vec(
                cluster=cluster,
                layer_idx=layer_idx,
                device=device,
                out_dtype=q_vec.dtype,
            )
            if c_vec.numel() == q_vec.numel() and c_vec.numel() > 0:
                vecs.append(c_vec)
                raw_idx.append(int(i))
        if len(vecs) == 0:
            return [], []

        cluster_mat = torch.stack(vecs, dim=0)
        scores = torch.matmul(cluster_mat, q_vec)
        if temp > 0:
            scores = scores / float(temp)
        topk_cfg = max(0, int(self.config.get("long_cluster_topk", 4)))
        k = min(int(topk_cfg), int(scores.numel()))
        if k <= 0:
            return [], []

        top_local = torch.topk(scores, k=k, largest=(score_order == "highest")).indices
        selected_cluster_idx = [raw_idx[int(x)] for x in top_local.detach().cpu().tolist()]
        selected_scores = scores.index_select(0, top_local).detach().float().cpu().tolist()
        return [int(x) for x in selected_cluster_idx], [float(x) for x in selected_scores]

    def _extract_video_features_rekv(
        self,
        frames: np.ndarray,
        device: str,
        prepared_pixel_values: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Return projected LLM video tokens.
        if frames.size == 0:
            return torch.empty(0, self.config["n_frame_tokens"], 0, dtype=torch.float16, device=device)

        if hasattr(self.model, "get_vision_tower"):
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower = vision_tower.to(device=device, dtype=torch.float16)
        else:
            vision_tower = self.model.vision_tower
            vision_tower = vision_tower.to(device=device, dtype=torch.float16)

        batch_size = self.config["vision_batch_size"]
        expected_batches = (len(frames) + batch_size - 1) // batch_size
        if prepared_pixel_values is not None and len(prepared_pixel_values) != expected_batches:
            raise ValueError(
                "Prepared OneVision visual-batch count mismatch: "
                f"expected={expected_batches}, actual={len(prepared_pixel_values)}"
            )
        feats_list: List[torch.Tensor] = []
        self._dbg(f"extract_video_features_rekv: total_frames={len(frames)} batch_size={batch_size}")
        self._dbg_mem("extract_video_features:start", device)

        with torch.inference_mode():
            for batch_index, i in enumerate(range(0, len(frames), batch_size)):
                chunk = frames[i : i + batch_size]
                self._dbg(f"extract_video_features_rekv: batch={i}-{i + len(chunk) - 1}")
                if prepared_pixel_values is None:
                    video_inputs = self.processor.video_processor(chunk, return_tensors="pt")
                    pixel_values = getattr(video_inputs, "pixel_values_videos", None)
                    if pixel_values is None:
                        raise RuntimeError("video_processor did not return pixel_values_videos")
                else:
                    pixel_values = prepared_pixel_values[batch_index]
                if pixel_values.ndim == 4:
                    pixel_values = pixel_values.unsqueeze(0)
                if pixel_values.ndim != 5:
                    raise RuntimeError(f"Unexpected pixel_values_videos ndim={pixel_values.ndim}")

                b, t, c, h, w = pixel_values.shape
                if int(b * t) != len(chunk):
                    raise ValueError(
                        "Prepared OneVision visual-frame count mismatch: "
                        f"batch={batch_index}, expected={len(chunk)}, actual={int(b * t)}"
                    )
                pixel_values = pixel_values.view(b * t, c, h, w).to(device=device, dtype=torch.float16)

                vision_out = vision_tower(pixel_values, output_hidden_states=True)
                if hasattr(vision_out, "hidden_states") and vision_out.hidden_states is not None:
                    layer_idx = getattr(self.model.config, "vision_feature_layer", -1)
                    selected = vision_out.hidden_states[layer_idx]
                else:
                    selected = vision_out.last_hidden_state if hasattr(vision_out, "last_hidden_state") else vision_out[0]

                strategy = getattr(self.model.config, "vision_feature_select_strategy", "default")
                if strategy == "default" and selected.shape[1] > 1:
                    selected = selected[:, 1:]

                if hasattr(self.model, "mm_projector"):
                    selected = self.model.mm_projector(selected)
                elif hasattr(self.model, "multi_modal_projector"):
                    selected = self.model.multi_modal_projector(selected)

                if hasattr(self.model, "apply_pooling"):
                    selected = self.model.apply_pooling(selected)  # (t, 196, hidden)

                feats_list.append(selected)
                self._dbg_mem("extract_video_features:after_batch", device)

        feats = torch.cat(feats_list, dim=0)  # (num_frames, 196, hidden)
        self._dbg(f"extract_video_features_rekv: shape={tuple(feats.shape)}")
        self._dbg_mem("extract_video_features:end", device)
        return feats
