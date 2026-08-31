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
from src.shallowstream.evidence_retrieval import validate_evidence_retrieval_config
from src.shallowstream.common import (
    SingleLayerLegacyCache,
    expand_temporal_neighbors,
    get_active_attn_implementation,
)
from src.utils.eval_io import atomic_write_json
from src.utils.streamingbench_decode_cache import acknowledge_decoded_cache
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


class OneVisionStreamingMixin:
    def _release_fullkv_prefill_allocator_cache(self) -> bool:
        enabled = bool(self.config.get("full_kv_mode", False) and torch.cuda.is_available())
        if enabled:
            torch.cuda.empty_cache()
        return enabled

    def _ensure_model_loaded(self) -> None:
        self._apply_config_overrides_from_env()
        validate_evidence_retrieval_config(self.config, model_family="onevision")
        self._selected_generate_mode()
        if self.model is not None:
            return
        if not str(self.config.get("model_path", "")).strip():
            raise ValueError(f"{self.log_name} requires an explicit model_path configuration")
        self.processor = LlavaOnevisionProcessor.from_pretrained(self.config["model_path"])
        self.tokenizer = self.processor.tokenizer
        torch_dtype = torch.float16
        model_cfg = AutoConfig.from_pretrained(self.config["model_path"])
        if hasattr(model_cfg, "torch_dtype"):
            model_cfg.torch_dtype = torch_dtype
        text_cfg = getattr(model_cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "torch_dtype"):
            text_cfg.torch_dtype = torch_dtype
        model_kwargs = {
            "device_map": self.config["device_map"],
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "config": model_cfg,
        }
        attn_impl = self.config.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.config["model_path"],
            **model_kwargs,
        )
        if hasattr(self.model, "config"):
            self.model.config.torch_dtype = torch_dtype
            text_cfg = getattr(self.model.config, "text_config", None)
            if text_cfg is not None:
                text_cfg.torch_dtype = torch_dtype
        if getattr(self.model.config, "image_aspect_ratio", None) is None:
            self.model.config.image_aspect_ratio = "pad"
        if attn_impl == "flash_attention_2":
            active = self._get_active_attn_implementation()
            if active != "flash_attention_2":
                raise RuntimeError(
                    f"{self.log_name} requires flash_attention_2, but active attention is '{active}'."
                )
        self.model.eval()
        print(f"[{self.log_name} Config] {self.config}")
        print(
            f"[{self.log_name}] attention implementation: "
            f"{self._get_active_attn_implementation()}",
            flush=True,
        )
        if self.config.get("debug_similarity"):
            out_dir = os.path.abspath(str(self.config.get("debug_similarity_dir", "./outputs/streamingbench/debug/retrieval")))
            os.makedirs(out_dir, exist_ok=True)
            print(
                f"[{self.log_name}] debug dump_dir (heatmap + selected_grid): {out_dir}",
                flush=True,
            )

    def _run_stream_step(
        self,
        file: str,
        audio_path: Optional[str],
        session_id: Optional[str],
        is_begin: bool,
        question_text: str,
        stage_recorder: Optional[StageRecorder] = None,
        prepared_video: Optional[Any] = None,
        task_gate_text: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._ensure_model_loaded()
        selected_generate_mode = self._selected_generate_mode()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device_obj)
        run_total_token = stage_recorder.start("pipeline_total") if stage_recorder else None

        reset = (
            is_begin
            or self.state["session"] != session_id
            or self.state["sample_fps"] != self.config["sample_fps"]
            or self.state["max_frames_num"] != self.config["max_frames_num"]
            or self.state["lower_kv"] is None
        )
        if reset:
            reset_token = stage_recorder.start("pipeline_reset_state") if stage_recorder else None
            self._reset_stream_state(
                session_id=session_id,
                sample_fps=self.config["sample_fps"],
                max_frames_num=self.config["max_frames_num"],
                device=device,
            )
            if torch.cuda.is_available():
                # A fresh OVO prefix no longer references the previous sample's
                # large attention workspaces. Return those segments before the
                # timed prefill so long-video cases do not inherit allocator
                # fragmentation from the preceding case.
                torch.cuda.empty_cache()
            if reset_token is not None:
                stage_recorder.end(reset_token)

        no_new_video_chunk = str(file) == self.no_new_video_chunk
        prefill_stats: Dict[str, Any] = {
            "frame_count": 0,
            "feature_extract_ms": 0.0,
            "visual_prefill_ms": 0.0,
            "caption_tokenize_ms": 0.0,
            "caption_prefill_ms": 0.0,
            "caption_token_count": 0,
            "long_promoted_frames": 0,
            "long_cluster_count": int(len(self.state.get("long_clusters") or [])),
            "total_ms": 0.0,
            "no_new_video_chunk": bool(no_new_video_chunk),
        }

        t_start = time.perf_counter()
        if not no_new_video_chunk:
            load_token = stage_recorder.start("video_frame_load") if stage_recorder else None
            frames, sampled_source_ids, sampled_fps = self._load_video_frames(
                file,
                sample_fps=self.config["sample_fps"],
                max_frames_num=self.config["max_frames_num"],
                prepared_video=prepared_video,
            )
            if load_token is not None:
                stage_recorder.end(load_token, frame_count=int(frames.shape[0]))
            self._dbg(f"stream_step: loaded_frames={frames.shape}")
            clip_start_s = self._infer_clip_start_seconds(file)
            if sampled_fps is not None and sampled_fps > 0:
                sampled_source_ts = [clip_start_s + float(idx) / float(sampled_fps) for idx in sampled_source_ids]
            else:
                sampled_source_ts = [clip_start_s for _ in sampled_source_ids]
            self._dbg_frames(
                f"loaded source_idx={sampled_source_ids} "
                f"source_ts_s={[round(float(x), 3) for x in sampled_source_ts]}"
            )
            self._dbg_mem("stream_step:after_load_frames", device)

            audio_token = stage_recorder.start("subtitle_transcription_or_cache") if stage_recorder else None
            self._update_audio_segments_for_chunk(file, audio_path, clip_start_s)
            if audio_token is not None:
                stage_recorder.end(audio_token, audio_segments_total=len(self.state.get("audio_segments") or []))

            append_token = stage_recorder.start("video_frame_prefill_kv") if stage_recorder else None
            attn_before_video_prefill = self._get_lower_attn_path_stats_copy()
            prefill_stats = self._append_video_chunk_prefill_lower(
                frames,
                sampled_source_ts,
                device,
                prepared_video=prepared_video,
            )
            attn_after_video_prefill = self._get_lower_attn_path_stats_copy()
            prefill_stats["lower_attn_path_stats_delta"] = self._diff_attn_stats(attn_after_video_prefill, attn_before_video_prefill)
            if append_token is not None:
                stage_recorder.end(
                    append_token,
                    frame_count=int(prefill_stats.get("frame_count", 0)),
                    visual_prefill_ms=float(prefill_stats.get("visual_prefill_ms", 0.0)),
                    caption_tokenize_ms=float(prefill_stats.get("caption_tokenize_ms", 0.0)),
                    caption_prefill_ms=float(prefill_stats.get("caption_prefill_ms", 0.0)),
                    caption_token_count=int(prefill_stats.get("caption_token_count", 0)),
                )

            caption_token = stage_recorder.start("subtitle_caption_alignment") if stage_recorder else None
            self._refresh_frame_captions_from_audio()
            if caption_token is not None:
                stage_recorder.end(caption_token, caption_count=len(self.state.get("frame_captions") or []))
            self._dbg_mem("stream_step:after_append_video", device)
        else:
            if stage_recorder is not None:
                load_token = stage_recorder.start("video_frame_load")
                stage_recorder.end(load_token, frame_count=0, skipped_no_new_video_chunk=True)
                audio_token = stage_recorder.start("subtitle_transcription_or_cache")
                stage_recorder.end(audio_token, audio_segments_total=len(self.state.get("audio_segments") or []), skipped_no_new_video_chunk=True)
                append_token = stage_recorder.start("video_frame_prefill_kv")
                stage_recorder.end(append_token, frame_count=0, skipped_no_new_video_chunk=True)
                caption_token = stage_recorder.start("subtitle_caption_alignment")
                stage_recorder.end(caption_token, caption_count=len(self.state.get("frame_captions") or []), skipped_no_new_video_chunk=True)

        self._write_cluster_size_debug(
            session_id=session_id,
            chunk_file=file,
            no_new_video_chunk=no_new_video_chunk,
        )

        if bool(self.config.get("latency_sync_cuda")) and torch.cuda.is_available():
            torch.cuda.synchronize(device_obj)
        # FullKV retains the live KV tensors but no longer needs temporary
        # prefill allocations. Returning unused allocator segments before
        # question prefill avoids fragmentation on long videos.
        prefill_query_cache_released = self._release_fullkv_prefill_allocator_cache()
        question_start = time.perf_counter()
        prefill_wall_ms = (question_start - t_start) * 1000.0
        prefill_token = stage_recorder.start("question_prefill_kv") if stage_recorder else None
        generation_prompt = question_text + str(self.config.get("assistant_suffix", ""))
        attn_before_qprefill = self._get_lower_attn_path_stats_copy()
        q_layer_vecs, hidden_after_prune, lower_with_prompt, prompt_ids = self._forward_question_once_for_retrieval_and_prefill(
            question_text=question_text,
            generation_prompt=generation_prompt,
            device=device,
            collect_all_layers=(
                bool(self.config.get("debug_similarity"))
                or self._retrieval_score_strategy() == "shallow_layer_token_vote"
            ),
        )
        attn_after_qprefill = self._get_lower_attn_path_stats_copy()
        if prefill_token is not None:
            stage_recorder.end(prefill_token, prompt_tokens=int(prompt_ids.shape[1]))

        select_token = stage_recorder.start("retrieve_relevant_kv") if stage_recorder else None
        debug_video_path = None if no_new_video_chunk else file
        selected_frames = self._select_frames_by_question(
            question_text,
            device,
            q_layer_vecs=q_layer_vecs,
            video_path=debug_video_path,
            task_gate_text=task_gate_text,
        )
        last_sel = self.state.get("last_selection")
        selected_long_cluster_ids = []
        if isinstance(last_sel, dict):
            selected_long_cluster_ids = [int(x) for x in (last_sel.get("long_cluster_indices") or [])]
        self._dump_selected_frame_images(question_text=question_text, selected_frames=selected_frames, video_path=debug_video_path)
        selected_audio_context = self._build_selected_audio_context(selected_frames)
        if selected_audio_context:
            self._dbg_frames(f"selected_audio_context={selected_audio_context}")
        if select_token is not None:
            stage_recorder.end(
                select_token,
                selected_frame_count=len(selected_frames),
                selected_long_cluster_count=len(selected_long_cluster_ids),
            )

        cache_token = stage_recorder.start("build_answer_decode_kv") if stage_recorder else None
        if selected_generate_mode == "internal_kv":
            lower_sel = self._build_answer_decode_lower_cache(
                selected_frames=selected_frames,
                selected_long_cluster_ids=selected_long_cluster_ids,
                lower_with_prompt=lower_with_prompt,
                prompt_len=int(prompt_ids.shape[1]),
                device=device,
            )
        else:
            lower_sel = {}
            # simple_prompt re-encodes only selected evidence and never consumes
            # the full query-appended cache after retrieval.
            lower_with_prompt = {}
            hidden_after_prune = None
            q_layer_vecs = None
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
        if cache_token is not None:
            stage_recorder.end(cache_token)
        retrieval_elapsed_ms = (time.perf_counter() - question_start) * 1000.0
        self._dbg_mem("stream_step:after_build_selected_cache", device)

        decode_token = stage_recorder.start("decode_answer_generation") if stage_recorder else None
        attn_before_decode = self._get_lower_attn_path_stats_copy()
        upper_flash_before = int(self.state.get("upper_flash_layer_calls", 0))
        upper_nonflash_before = int(self.state.get("upper_nonflash_layer_calls", 0))
        if selected_generate_mode == "simple_prompt":
            response, decode_ttft_ms, decode_total_ms, generated_tokens = (
                self._decode_from_selected_prompt(
                    prompt_ids=prompt_ids,
                    selected_frames=selected_frames,
                    selected_long_cluster_ids=selected_long_cluster_ids,
                    device=device,
                )
            )
        else:
            response, decode_ttft_ms, decode_total_ms, generated_tokens = self._decode_from_reused_shallow_prefill(
                prompt_ids=prompt_ids,
                hidden_after_prune=hidden_after_prune,
                selected_frames=selected_frames,
                selected_long_cluster_ids=selected_long_cluster_ids,
                selected_lower_with_prompt_kv=lower_sel,
                device=device,
            )
        if bool(self.config.get("latency_sync_cuda")) and torch.cuda.is_available():
            torch.cuda.synchronize(device_obj)
        query_output_ms = (time.perf_counter() - question_start) * 1000.0
        attn_after_decode = self._get_lower_attn_path_stats_copy()
        upper_flash_after = int(self.state.get("upper_flash_layer_calls", 0))
        upper_nonflash_after = int(self.state.get("upper_nonflash_layer_calls", 0))
        if decode_token is not None:
            stage_recorder.end(
                decode_token,
                decode_ttft_ms=float(decode_ttft_ms),
                decode_total_ms=float(decode_total_ms),
                generated_tokens=int(generated_tokens),
            )

        post_token = stage_recorder.start("postprocess_output") if stage_recorder else None
        ttft_ms = retrieval_elapsed_ms + float(decode_ttft_ms)
        raw_response = response
        if "Options:" in question_text and "best option" in question_text.lower():
            letter = self._extract_choice_letter(response)
            if letter:
                response = letter
        if post_token is not None:
            stage_recorder.end(post_token, normalized_output=bool(raw_response != response))

        metrics: Dict[str, Any] = {
            "ttft_ms": float(ttft_ms),
            "decode_ttft_ms": float(decode_ttft_ms),
            "decode_total_ms": float(decode_total_ms),
            "answer_generation_ms": max(0.0, float(decode_total_ms - decode_ttft_ms)),
            "retrieval_ms": float(retrieval_elapsed_ms),
            "prefill_wall_ms": float(prefill_wall_ms),
            "query_output_ms": float(query_output_ms),
            "video_duration_s": self.state.get("_last_video_duration_s"),
            "sample_fps": float(self.config["sample_fps"]),
            "max_frames_num": self.config.get("max_frames_num"),
            "sampled_frames": int(prefill_stats.get("frame_count", 0)),
            "latency_ms": (time.perf_counter() - t_start) * 1000.0,
            "audio_prefill_mode": "frame_caption_kv",
            "audio_segments_total": len(self.state.get("audio_segments") or []),
            "selected_frame_count": len(selected_frames) + len(selected_long_cluster_ids),
            "selected_short_frame_count": len(selected_frames),
            "selected_long_cluster_count": len(selected_long_cluster_ids),
            "long_cluster_count_total": len(self.state.get("long_clusters") or []),
            "generated_tokens": int(generated_tokens),
            "prefill_stats": prefill_stats,
            "no_new_video_chunk": bool(no_new_video_chunk),
            "memory_policy": (
                "full_kv" if bool(self.config.get("full_kv_mode", False)) else "shallowstream"
            ),
            "full_kv_mode": bool(self.config.get("full_kv_mode", False)),
            "selected_generate_mode": selected_generate_mode,
            "prefill_layer_count": int(self.config.get("prune_layer", 0)),
            "prefill_query_cache_released": prefill_query_cache_released,
            "task_gate_decision": dict(self.state.get("last_gate_decision") or {}),
        }
        if selected_generate_mode == "simple_prompt":
            metrics["simple_prompt_stats"] = dict(
                self.state.get("last_simple_prompt_stats") or {}
            )
        metrics["lower_attn_path_stats"] = self._get_lower_attn_path_stats_copy()
        metrics["question_prefill_lower_attn_paths"] = self._diff_attn_stats(attn_after_qprefill, attn_before_qprefill)
        metrics["decode_lower_attn_paths"] = self._diff_attn_stats(attn_after_decode, attn_before_decode)
        metrics["upper_attn_impl"] = self._get_active_attn_implementation()
        metrics["upper_attn_layer_calls"] = {
            "flash": int(upper_flash_after),
            "nonflash": int(upper_nonflash_after),
            "decode_flash_delta": int(upper_flash_after - upper_flash_before),
            "decode_nonflash_delta": int(upper_nonflash_after - upper_nonflash_before),
        }
        if selected_audio_context:
            metrics["selected_audio_context"] = selected_audio_context
        if raw_response != response:
            metrics["raw_output"] = raw_response
        if torch.cuda.is_available():
            metrics["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated(device=device_obj) / (1024 ** 2)
            metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device=device_obj) / (1024 ** 2)
            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device=device_obj) / (1024 ** 2)

        if run_total_token is not None:
            stage_recorder.end(run_total_token)
        if stage_recorder is not None:
            phase_durations_ms = stage_recorder.durations_ms()
            metrics["phase_durations_ms"] = phase_durations_ms
            metrics["stage_timeline"] = stage_recorder.stages()
            metrics["timing_ms"] = {
                "pipeline_total_ms": float(phase_durations_ms.get("pipeline_total", metrics["latency_ms"])),
                "video_frame_load_ms": float(phase_durations_ms.get("video_frame_load", 0.0)),
                "subtitle_transcription_or_cache_ms": float(phase_durations_ms.get("subtitle_transcription_or_cache", 0.0)),
                "video_frame_prefill_kv_ms": float(phase_durations_ms.get("video_frame_prefill_kv", 0.0)),
                "video_visual_prefill_ms": float(prefill_stats.get("visual_prefill_ms", 0.0)),
                "subtitle_caption_tokenize_ms": float(prefill_stats.get("caption_tokenize_ms", 0.0)),
                "subtitle_caption_prefill_ms": float(prefill_stats.get("caption_prefill_ms", 0.0)),
                "subtitle_caption_alignment_ms": float(phase_durations_ms.get("subtitle_caption_alignment", 0.0)),
                "question_prefill_kv_ms": float(phase_durations_ms.get("question_prefill_kv", 0.0)),
                "retrieve_relevant_kv_ms": float(phase_durations_ms.get("retrieve_relevant_kv", 0.0)),
                "build_answer_decode_kv_ms": float(phase_durations_ms.get("build_answer_decode_kv", 0.0)),
                "decode_answer_generation_ms": float(phase_durations_ms.get("decode_answer_generation", decode_total_ms)),
                "decode_first_token_ms": float(decode_ttft_ms),
                "decode_after_first_token_ms": max(0.0, float(decode_total_ms - decode_ttft_ms)),
                "postprocess_output_ms": float(phase_durations_ms.get("postprocess_output", 0.0)),
            }
            metrics["timing_ms"]["subtitle_total_ms"] = (
                metrics["timing_ms"]["subtitle_transcription_or_cache_ms"]
                + metrics["timing_ms"]["subtitle_caption_tokenize_ms"]
                + metrics["timing_ms"]["subtitle_caption_prefill_ms"]
                + metrics["timing_ms"]["subtitle_caption_alignment_ms"]
            )
            metrics["timing_ms"]["retrieval_total_ms"] = (
                metrics["timing_ms"]["question_prefill_kv_ms"]
                + metrics["timing_ms"]["retrieve_relevant_kv_ms"]
                + metrics["timing_ms"]["build_answer_decode_kv_ms"]
            )

        self._dbg_mem("stream_step:end", device)
        if not no_new_video_chunk:
            acknowledge_decoded_cache(file)
        return response, metrics

    def _run_single(self, file: str, question_text: str) -> str:
        response, _ = self._run_stream_step(
            file=file,
            audio_path=None,
            session_id="single_run",
            is_begin=True,
            question_text=question_text,
        )
        return response
