from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from src.config import apply_config_sources
from src.modelclass import Model
from src.utils.time_trace import (
    StageRecorder,
    TimeTraceWriter,
    extract_clip_range_seconds,
    extract_sample_id,
    make_prompt_preview,
)

from .attention import OneVisionAttentionMixin
from .config import (
    LONG_CLUSTER_COSINE_SIM_THRESHOLD,
    NO_NEW_VIDEO_CHUNK,
    ONEVISION_V3_DEFAULT_CONFIG,
    new_runtime_state,
)
from .debug import OneVisionDebugMixin
from .decode import OneVisionDecodeMixin
from .history_decay import OneVisionHistoryDecayMixin
from .media import OneVisionMediaMixin
from .memory import OneVisionMemoryMixin
from .prefill import OneVisionPrefillMixin
from .prompt_generation import OneVisionPromptGenerationMixin
from .retrieval import OneVisionRetrievalMixin
from .retrieval_scores import OneVisionRetrievalScoreMixin
from .streaming import OneVisionStreamingMixin
from .task_gate import OneVisionTaskGateMixin
from .token_vote import OneVisionTokenVoteMixin


class LLaVAOneVisionRuntime(
    OneVisionDebugMixin,
    OneVisionMediaMixin,
    OneVisionAttentionMixin,
    OneVisionHistoryDecayMixin,
    OneVisionTaskGateMixin,
    OneVisionTokenVoteMixin,
    OneVisionRetrievalScoreMixin,
    OneVisionRetrievalMixin,
    OneVisionMemoryMixin,
    OneVisionPrefillMixin,
    OneVisionPromptGenerationMixin,
    OneVisionDecodeMixin,
    OneVisionStreamingMixin,
):
    """Shared V3-family runtime composed from responsibility-focused mixins."""

    version = 0
    runtime_name = ""
    config_env_prefix = ""
    default_config_file = ""
    cluster_chunk_debug_dir = ""
    explicit_asr_cache_cause = False
    no_new_video_chunk = NO_NEW_VIDEO_CHUNK
    long_cluster_cosine_sim_threshold = LONG_CLUSTER_COSINE_SIM_THRESHOLD

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.state = new_runtime_state()
        self._asr_model = None
        self._asr_unavailable_reason: Optional[str] = None
        self._env_overrides_applied = False

    @property
    def log_name(self) -> str:
        return self.runtime_name or f"ShallowStreamV{self.version}"

    def _apply_config_overrides_from_env(self) -> None:
        if self._env_overrides_applied:
            return
        self._env_overrides_applied = True

        env_prefix = self.config_env_prefix or f"SHALLOWSTREAM_V{self.version}"
        loaded = apply_config_sources(
            self.config,
            name=self.log_name,
            default_file=self.default_config_file,
            file_envs=(f"{env_prefix}_CONFIG_FILE", f"{env_prefix}_CONFIG_OVERRIDE_FILE"),
            json_envs=(f"{env_prefix}_CONFIG_JSON", f"{env_prefix}_CONFIG_OVERRIDE_JSON"),
            print_prefix=f"[{self.log_name} Config Override]",
        )
        self.config.clear()
        self.config.update(loaded)


class ShallowStreamLLaVAOneVisionBase(Model):
    """Stable benchmark interface backed by a version-specific runtime adapter."""

    runtime: LLaVAOneVisionRuntime
    model_name = "ShallowStream-LLaVA-OneVision"
    supports_persistent_stream = True
    supports_no_new_video_chunk = True

    def __init__(
        self,
        enable_time_trace: bool = False,
        time_trace_dir: str = "./outputs/streamingbench/timing",
        time_trace_tag: str = "",
    ) -> None:
        self.enable_time_trace = bool(enable_time_trace)
        self._task_gate_question = ""
        self.time_trace_writer = TimeTraceWriter(
            enabled=self.enable_time_trace,
            output_dir=time_trace_dir,
            model_name=self.name(),
            run_tag=time_trace_tag,
        )
        self.runtime._ensure_model_loaded()
        if self.enable_time_trace and self.time_trace_writer.path:
            print(f"[{self.runtime.log_name} TimeTrace] enabled -> {self.time_trace_writer.path}")

    def ensure_model_loaded(self) -> None:
        """Load and validate the real model before worker readiness is published."""

        self.runtime._ensure_model_loaded()
        if self.runtime.model is None:
            raise RuntimeError(
                f"{self.runtime.log_name} model load returned without a model instance"
            )

    def set_task_gate_question(self, question: str) -> None:
        raw_question = str(question).strip()
        if not raw_question:
            raise ValueError("task gate question must not be empty")
        self._task_gate_question = raw_question

    def _consume_task_gate_question(self, explicit_question=None):
        if explicit_question is not None:
            raw_question = str(explicit_question).strip()
            if not raw_question:
                raise ValueError("task gate question must not be empty")
        else:
            raw_question = str(getattr(self, "_task_gate_question", "")).strip()
        self._task_gate_question = ""
        return raw_question or None

    def Run(self, file, inp):
        response, _metrics = self.runtime._run_stream_step(
            file=file,
            audio_path=None,
            session_id="single_run",
            is_begin=True,
            question_text=inp,
            task_gate_text=self._consume_task_gate_question(),
        )
        return response

    def Run_With_Metrics(
        self,
        file,
        inp,
        *,
        prepared_video=None,
        task_gate_text=None,
    ):
        return self.runtime._run_stream_step(
            file=file,
            audio_path=None,
            session_id="single_run",
            is_begin=True,
            question_text=inp,
            prepared_video=prepared_video,
            task_gate_text=self._consume_task_gate_question(task_gate_text),
        )

    def Run_Text_Stream(self, file, audio, session, isBegin, inp):
        stage_recorder = StageRecorder() if self.enable_time_trace else None
        response, metrics = self.runtime._run_stream_step(
            file=file,
            audio_path=audio,
            session_id=str(session) if session is not None else None,
            is_begin=bool(isBegin),
            question_text=inp,
            stage_recorder=stage_recorder,
            task_gate_text=self._consume_task_gate_question(),
        )

        if stage_recorder is not None:
            if self.time_trace_writer.path:
                metrics["time_trace_file"] = self.time_trace_writer.path

            trace_video_path = None if str(file) == self.runtime.no_new_video_chunk else file
            clip_start_s, clip_end_s = extract_clip_range_seconds(trace_video_path)
            trace_record = {
                "record_type": "sample_timing",
                "model": self.name(),
                "sample_id": extract_sample_id(trace_video_path, str(audio) if audio is not None else None),
                "session_id": str(session) if session is not None else None,
                "is_begin": bool(isBegin),
                "video_clip_path": (os.path.abspath(str(trace_video_path)) if trace_video_path else None),
                "audio_path": str(audio) if audio is not None else None,
                "clip_start_s": clip_start_s,
                "clip_end_s": clip_end_s,
                "prompt_preview": make_prompt_preview(inp),
                "recorded_at_ts": time.time(),
                "stages": stage_recorder.stages(),
                "phase_durations_ms": metrics.get("phase_durations_ms", {}),
                "timing_ms": metrics.get("timing_ms", {}),
                "prefill_stats": metrics.get("prefill_stats", {}),
                "attn_stats": {
                    "lower_attn_path_stats": metrics.get("lower_attn_path_stats", {}),
                    "question_prefill_lower_attn_paths": metrics.get("question_prefill_lower_attn_paths", {}),
                    "decode_lower_attn_paths": metrics.get("decode_lower_attn_paths", {}),
                    "upper_attn_impl": metrics.get("upper_attn_impl"),
                    "upper_attn_layer_calls": metrics.get("upper_attn_layer_calls", {}),
                },
                "metrics": {
                    key: metrics.get(key)
                    for key in [
                        "ttft_ms",
                        "decode_ttft_ms",
                        "decode_total_ms",
                        "answer_generation_ms",
                        "retrieval_ms",
                        "latency_ms",
                        "generated_tokens",
                        "gpu_peak_mem_mb",
                    ]
                    if key in metrics
                },
            }
            self.time_trace_writer.write(trace_record)

        return response, metrics

    def name(self):
        return self.model_name
