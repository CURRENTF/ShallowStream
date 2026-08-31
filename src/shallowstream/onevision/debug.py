from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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


class OneVisionDebugMixin:
    def _dbg(self, msg: str) -> None:
        if self.config.get("debug"):
            print(f"[{self.log_name}-Debug] {msg}", flush=True)

    def _dbg_frames(self, msg: str) -> None:
        if self.config.get("debug_frames"):
            print(f"[{self.log_name}-Frame] {msg}", flush=True)

    def _dbg_mem(self, stage: str, device: str) -> None:
        if not self.config.get("debug_mem") or not torch.cuda.is_available():
            return
        dev = torch.device(device)
        idx = dev.index if dev.type == "cuda" and dev.index is not None else torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
        free, total = torch.cuda.mem_get_info(idx)
        print(
            f"[{self.log_name}-MEM] {stage}: alloc={alloc:.1f}MB reserved={reserved:.1f}MB "
            f"max_alloc={max_alloc:.1f}MB free={free/(1024**2):.1f}MB total={total/(1024**2):.1f}MB",
            flush=True,
        )

    def _new_lower_attn_path_stats(self) -> Dict[str, int]:
        return {
            "flash_full_causal": 0,
            "flash_local_suffix": 0,
            "flash_sink_local_concat": 0,
        }

    def _local_window_tokens(self) -> int:
        if bool(self.config.get("full_kv_mode", False)):
            return 0
        frames = int(self.config.get("shallow_prefill_local_window_frames", 0))
        tokens_per_frame = int(self.config.get("n_frame_tokens", 0))
        if frames <= 0:
            raise ValueError("shallow_prefill_local_window_frames must be positive")
        if tokens_per_frame <= 0:
            raise ValueError("n_frame_tokens must be positive")
        return frames * tokens_per_frame

    def _local_tail_tokens(self, query_len: int) -> int:
        window_tokens = self._local_window_tokens()
        if window_tokens <= 0:
            return 0
        query_len = int(query_len)
        if query_len <= 0:
            raise ValueError("query_len must be positive")
        return window_tokens + query_len

    def _get_lower_attn_path_stats_copy(self) -> Dict[str, int]:
        stats = self.state.get("lower_attn_path_stats")
        if not isinstance(stats, dict):
            stats = self._new_lower_attn_path_stats()
        out: Dict[str, int] = {}
        for k, v in stats.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                out[str(k)] = 0
        for k in self._new_lower_attn_path_stats().keys():
            out.setdefault(k, 0)
        return out

    def _bump_lower_attn_path(self, key: str) -> None:
        stats = self.state.get("lower_attn_path_stats")
        if not isinstance(stats, dict):
            stats = self._new_lower_attn_path_stats()
            self.state["lower_attn_path_stats"] = stats
        stats[key] = int(stats.get(key, 0)) + 1

    def _diff_attn_stats(self, after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
        keys = set(before.keys()) | set(after.keys())
        return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in sorted(keys)}

    def _make_debug_thumb(self, frame_rgb: np.ndarray) -> np.ndarray:
        max_side = max(32, int(self.config.get("debug_grid_max_side", 320)))
        img = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        w, h = img.size
        if max(w, h) > max_side:
            if w >= h:
                nw = max_side
                nh = max(1, int(round(h * max_side / float(w))))
            else:
                nh = max_side
                nw = max(1, int(round(w * max_side / float(h))))
            img = img.resize((nw, nh), Image.BICUBIC)
        return np.asarray(img, dtype=np.uint8)

    def _video_debug_subdir_name(self, video_path: Optional[str]) -> str:
        base = os.path.splitext(os.path.basename(str(video_path or "unknown_video")))[0]
        # Collapse clip names like sample_1_real_4_53 -> sample_1_real
        m = re.match(r"(.+)_\d+(?:\.\d+)?_\d+(?:\.\d+)?$", base)
        if m:
            base = m.group(1)
        return self._slugify(base, max_len=96)

    def _dump_selected_frame_images(self, question_text: str, selected_frames: List[int], video_path: Optional[str]) -> None:
        if not self.config.get("debug_similarity"):
            return
        thumbs = self.state.get("frame_debug_thumbs")
        source_ts = self.state.get("frame_source_ids")
        if not isinstance(thumbs, list) or len(thumbs) == 0:
            self._dbg_frames("selected_grid skipped: no frame_debug_thumbs")
            return

        core_q = self._extract_core_question_text(question_text)
        # Keep grid and heatmap in the same per-question folder.
        qid = max(0, int(self.state.get("question_counter", 0)) - 1)
        stem = f"{qid:04d}_{self._slugify(core_q)}"
        out_root = os.path.abspath(str(self.config.get("debug_similarity_dir", "./outputs/streamingbench/debug/retrieval")))
        video_dir = self._video_debug_subdir_name(video_path)
        out_dir = os.path.join(out_root, video_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        selection = self.state.get("last_selection")
        score_map: Dict[int, float] = {}
        if isinstance(selection, dict):
            for idx, score in zip(selection.get("keep_idx", []), selection.get("selected_scores", [])):
                score_map[int(idx)] = float(score)

        selected = sorted(set(int(fid) for fid in selected_frames))
        tile_imgs: List[Image.Image] = []
        for rank, fid in enumerate(selected):
            if fid < 0 or fid >= len(thumbs):
                continue
            img = Image.fromarray(thumbs[fid].astype(np.uint8), mode="RGB")
            draw = ImageDraw.Draw(img)
            ts = float(source_ts[fid]) if isinstance(source_ts, list) and fid < len(source_ts) else -1.0
            score = score_map.get(fid, None)
            badge = f"#{rank} idx={fid} t={ts:.2f}s"
            if score is not None:
                badge += f" s={score:.4f}"
            draw.rectangle([0, 0, img.size[0], 20], fill=(0, 0, 0))
            draw.text((4, 3), badge, fill=(255, 255, 255))
            tile_imgs.append(img)

        if not tile_imgs:
            return

        cols = min(4, len(tile_imgs))
        rows_n = int(math.ceil(len(tile_imgs) / float(cols)))
        tw = max(im.size[0] for im in tile_imgs)
        th = max(im.size[1] for im in tile_imgs)
        canvas = Image.new("RGB", (cols * tw, rows_n * th), (245, 245, 245))
        for i, im in enumerate(tile_imgs):
            r = i // cols
            c = i % cols
            x = c * tw + (tw - im.size[0]) // 2
            y = r * th + (th - im.size[1]) // 2
            canvas.paste(im, (x, y))
        grid_path = os.path.join(out_dir, "selected_grid.jpg")
        canvas.save(grid_path, quality=95)
        self._dbg_frames(f"selected_grid dump={grid_path} frames={len(tile_imgs)}")

    def _slugify(self, text: str, max_len: int = 48) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
        if not s:
            s = "q"
        return s[:max_len]

    def _get_long_cluster_frame_counts(self) -> List[int]:
        clusters = self.state.get("long_clusters")
        if not isinstance(clusters, list):
            return []
        out: List[int] = []
        for c in clusters:
            try:
                out.append(max(0, int(c.get("count", 0))))
            except Exception:
                out.append(0)
        return out

    def _write_cluster_size_debug(
        self,
        session_id: Optional[str],
        chunk_file: str,
        no_new_video_chunk: bool,
    ) -> None:
        if not bool(self.config.get("long_cluster_debug", False)):
            return
        try:
            out_dir = os.path.abspath(self.cluster_chunk_debug_dir)
            os.makedirs(out_dir, exist_ok=True)
            sid = self._slugify(str(session_id) if session_id is not None else "default_session", max_len=80)
            out_path = os.path.join(out_dir, f"{sid}.jsonl")
            frame_counts = self._get_long_cluster_frame_counts()
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session": str(session_id) if session_id is not None else None,
                "chunk_file": str(chunk_file),
                "no_new_video_chunk": bool(no_new_video_chunk),
                "cluster_count": int(len(frame_counts)),
                "cluster_frame_counts": [int(x) for x in frame_counts],
                "cluster_total_frames": int(sum(frame_counts)),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._dbg(f"cluster_size_debug write failed: {exc}")

    def _dump_similarity_debug(
        self,
        question_text: str,
        scores: torch.Tensor,
        source_ts: List[float],
        keep_idx: List[int],
        video_path: Optional[str] = None,
        recent_idx: Optional[List[int]] = None,
        important_seed_idx: Optional[List[int]] = None,
        important_expanded_idx: Optional[List[int]] = None,
        layer_scores: Optional[List[List[float]]] = None,
        score_order: str = "highest",
    ) -> None:
        if not self.config.get("debug_similarity"):
            return

        core_q = self._extract_core_question_text(question_text)
        qid = int(self.state.get("question_counter", 0))
        self.state["question_counter"] = qid + 1
        stem = f"{qid:04d}_{self._slugify(core_q)}"
        out_root = os.path.abspath(str(self.config.get("debug_similarity_dir", "./outputs/streamingbench/debug/retrieval")))
        video_dir = self._video_debug_subdir_name(video_path)
        out_dir = os.path.join(out_root, video_dir, stem)
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, "similarity_heatmap.png")

        score_list = scores.detach().float().cpu().tolist()
        n = len(score_list)
        selected_set = set(int(x) for x in keep_idx)
        recent_set = set(int(x) for x in (recent_idx or []))
        important_seed_set = set(int(x) for x in (important_seed_idx or []))
        important_expanded_set = set(int(x) for x in (important_expanded_idx or []))

        cell_w = 10
        title_h = 22
        axis_h = 18
        def _draw_time_axis(draw: ImageDraw.ImageDraw, axis_y: int, frames_n: int) -> None:
            if frames_n <= 0:
                return
            src = [float(source_ts[i]) if i < len(source_ts) else -1.0 for i in range(frames_n)]
            # Show around 8 ticks to avoid clutter.
            step = max(1, int(math.ceil(frames_n / 8.0)))
            for i in range(0, frames_n, step):
                x0 = i * cell_w
                draw.line([x0, axis_y, x0, axis_y + 3], fill=(30, 30, 30), width=1)
                draw.text((x0 + 1, axis_y + 4), f"{src[i]:.1f}s", fill=(20, 20, 20))
            x_last = (frames_n - 1) * cell_w
            draw.line([x_last, axis_y, x_last, axis_y + 3], fill=(30, 30, 30), width=1)
            draw.text((x_last + 1, axis_y + 4), f"{src[-1]:.1f}s", fill=(20, 20, 20))

        if layer_scores is not None and len(layer_scores) > 0:
            layers_n = len(layer_scores)
            frames_n = len(layer_scores[0]) if len(layer_scores[0]) > 0 else n
            cell_h = 12
            width = max(1, frames_n) * cell_w
            height = title_h + max(1, layers_n) * cell_h + axis_h
            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((2, 1), f"order={score_order} norm=per_layer", fill=(0, 0, 0))
            for l in range(layers_n):
                row = layer_scores[l]
                if len(row) == 0:
                    continue
                row_min = min(row)
                row_max = max(row)
                row_denom = (row_max - row_min) if (row_max - row_min) > 1e-12 else 1.0
                for i, s in enumerate(row):
                    v = (float(s) - row_min) / row_denom
                    r = int(255 * v)
                    g = int(32 * (1.0 - abs(v - 0.5) * 2.0))
                    b = int(255 * (1.0 - v))
                    x0 = i * cell_w
                    x1 = x0 + cell_w - 1
                    y0 = title_h + l * cell_h
                    y1 = y0 + cell_h - 1
                    draw.rectangle([x0, y0, x1, y1], fill=(r, g, b))
            # Overlay selected/recent/important markers by column.
            for i in range(frames_n):
                x0 = i * cell_w
                x1 = x0 + cell_w - 1
                if i in recent_set:
                    draw.rectangle([x0, 14, x1, 16], fill=(66, 133, 244))  # blue: recent
                if i in important_expanded_set:
                    draw.rectangle([x0, 17, x1, 19], fill=(255, 153, 0))   # orange: expanded
                if i in important_seed_set:
                    draw.rectangle([x0, 20, x1, 21], fill=(220, 0, 0))     # red: seed
                if i in selected_set:
                    draw.rectangle([x0, title_h, x1, title_h + max(1, layers_n) * cell_h - 1], outline=(0, 255, 0), width=1)
            _draw_time_axis(draw, title_h + max(1, layers_n) * cell_h, frames_n)
        else:
            width = max(1, n) * cell_w
            height = title_h + 80 + axis_h
            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((2, 1), f"order={score_order}", fill=(0, 0, 0))
            if n > 0:
                s_min = min(score_list)
                s_max = max(score_list)
                denom = (s_max - s_min) if (s_max - s_min) > 1e-12 else 1.0
                for i, s in enumerate(score_list):
                    v = (s - s_min) / denom
                    r = int(255 * v)
                    g = int(32 * (1.0 - abs(v - 0.5) * 2.0))
                    b = int(255 * (1.0 - v))
                    x0 = i * cell_w
                    x1 = x0 + cell_w - 1
                    draw.rectangle([x0, title_h, x1, height - 1], fill=(r, g, b))
                    if i in recent_set:
                        draw.rectangle([x0, 14, x1, 16], fill=(66, 133, 244))
                    if i in important_expanded_set:
                        draw.rectangle([x0, 17, x1, 19], fill=(255, 153, 0))
                    if i in important_seed_set:
                        draw.rectangle([x0, 20, x1, 21], fill=(220, 0, 0))
                    if i in selected_set:
                        draw.rectangle([x0, title_h, x1, title_h + 80 - 1], outline=(0, 255, 0), width=1)
                _draw_time_axis(draw, title_h + 80, n)
        img.save(png_path)

        self._dbg_frames(f"similarity_heatmap dump={png_path}")
