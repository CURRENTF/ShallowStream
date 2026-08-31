"""Configuration and scalar helpers for ShallowStream Qwen3-VL V3."""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from src.config import apply_config_sources
from src.shallowstream.common import get_active_attn_implementation as _get_active_attn_implementation


MODEL_NAME = "ShallowStream_Qwen3VL_V3"

DEFAULT_TASK_GATE_PAST_ANCHOR = (
    "This question is about something that happened earlier in the video. "
    "It asks the model to recall a past action, past object state, past interaction, "
    "or something that was seen before but may not be visible now. "
    "It may also require remembering repeated events across time, accumulating history, "
    "or recognizing a counting question such as how many times something happened. "
    "The question is usually phrased in the past tense or requires retrieval from earlier visual memory."
)
DEFAULT_TASK_GATE_NONPAST_ANCHOR = (
    "This question is not about retrieving earlier visual memory. "
    "It asks what is happening now, what is visible in the current scene, "
    "or what will happen next or later. "
    "Its evidence is in the current frames or future frames rather than in events before the current moment."
)
DEFAULT_CONFIG: Dict[str, object] = {
    "model_path": "",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "attn_implementation": "flash_attention_2",
    # Baseline-only mode. The adapter resolves prune_layer to the complete
    # decoder depth and the runtime enforces global attention/full retention.
    "full_kv_mode": False,
    # StreamingBench asks multiple ordered questions per stream. The OVO
    # FullKV path may consume its one-shot visual cache after answering, while
    # this opt-in keeps an exact CPU-resident visual KV archive between
    # questions and restores it for the next incremental append.
    "fullkv_preserve_stream_history": False,
    "sample_fps": 1.0,
    # 0 means do not cap Qwen's fps-based sampled frames.
    "max_sampled_frames": 0,
    # Positive values request exactly this many uniformly spaced frames. The
    # decoder repeats indices when the source contains fewer raw frames. This
    # is mutually exclusive with max_sampled_frames and ignores sample_fps.
    "exact_uniform_sampled_frames": 0,
    # 0 means let decord choose its default thread policy.
    "video_decode_threads": 0,
    # Qwen3-VL video tokens are grouped by temporal_patch_size sampled frames
    # into one video unit. Retrieval, clustering, and eviction below operate on
    # those video units, not raw sampled frames.
    "retrieval_recent_units": 4,
    "long_cluster_topk": 4,
    # This switch controls construction as well as retrieval. A zero top-k
    # only means "retrieve none" and is not an off switch for cluster work.
    "long_cluster_enabled": True,
    # Match ShallowStream-LLaVA V3: start a new long cluster when cosine similarity
    # to the latest cluster center falls below this threshold.
    "cluster_threshold": 0.99,
    "cluster_center_image_size": 336,
    "frame_embedding_size": 32,
    "frame_max_pixels": 360 * 420,
    "video_total_pixels": 24576 * 32 * 32,
    # qwen-vl-utils defaults to patch size 14, but Qwen3-VL HF video
    # processor uses 16.  Keep these aligned so fetch_video output can be
    # consumed with do_resize=False.
    "video_image_patch_size": 16,
    "video_temporal_patch_size": 2,
    # Optional exact-value cache for qwen_vl_utils.fetch_video outputs. Disabled
    # by default so normal runs still call fetch_video directly.
    "fetch_video_cache_dir": "",
    # SimpleStream may ask the shared producer to retain only the final
    # temporal chunks; zero preserves full-history ShallowStream behavior.
    "fetch_video_recent_frames": 0,
    # original preserves fetch_video's tensor dtype. uint8 is an opt-in exact
    # compression for integer-valued [0, 255] video tensors and fails loudly if
    # a future decoder returns values that cannot round-trip exactly.
    "fetch_video_cache_storage_dtype": "original",
    # Legacy frame descriptors are not used by the active Q/K retrieval path.
    # They can be disabled and are computed lazily if build_memory_frames is
    # explicitly called.
    "compute_frame_embeddings": True,
    # When true, existing fetch_video cache entries are used but misses are not
    # written. This is useful for large evaluations when the cache filesystem is
    # close to full.
    "fetch_video_cache_read_only": False,
    "fetch_video_cache_write_strict": False,
    # Cache consumers normally decode a miss for backward compatibility. A
    # producer/consumer sweep sets this to ``wait`` so GPU workers never repeat
    # CPU video decoding while the producer is filling the shared cache.
    "fetch_video_cache_miss_policy": "decode",
    "fetch_video_cache_wait_timeout_seconds": 1800.0,
    "fetch_video_cache_wait_poll_seconds": 0.1,
    # Optional routed-cache acknowledgement channel. Both values are set
    # together by the evaluation worker; normal model use leaves them disabled.
    "fetch_video_cache_route_state_dir": "",
    "fetch_video_cache_consumer_id": "",
    # Sharded evaluators may reuse one decoded entry for multiple sample cases.
    # The worker retains one local decoded copy before acknowledging the shared
    # routed entry, then releases that copy after its final planned local use.
    "fetch_video_cache_consumer_use_counts": {},
    "max_new_tokens": 16,
    # Benchmark-only fixed-work decode. Normal quality runs should leave this
    # disabled so EOS retains its usual semantics.
    "force_exact_new_tokens": False,
    "latency_sync_cuda": False,
    # Systems-only reference mode: measure the post-decode streaming state
    # update, then stop before query/generation allocations. This is useful
    # when a full-depth retained history fills the device before answering.
    "latency_prefill_only": False,
    # Systems-only history scaling. 0 preserves the complete decoded sample;
    # a positive value keeps that exact even-length decoded prefix so Qwen's
    # temporal patching and the frame/metadata arrays remain aligned.
    "latency_history_prefix_frames": 0,
    # Opt-in systems profiling. Empty checkpoints preserve the production
    # latency path without allocator synchronization or memory snapshots.
    "memory_profile_checkpoints_frames": [],
    "memory_profile_empty_cache_before_sample": False,
    # HERMES baseline controls. These live in the shared Qwen decode config so
    # the unified OVO producer can keep using one decode/cache contract while
    # the evaluator swaps only the GPU-side memory policy.
    "hermes_kv_size": 4096,
    "hermes_encode_chunk_frames": 16,
    "hermes_short_term_ratio": 0.1,
    "hermes_long_term_ratio": 0.3,
    "hermes_repetition_penalty": 1.1,
    "hermes_enabled": False,
    "do_sample": False,
    "decode_temperature": 1.0,
    "debug": False,
    # OVO-Bench binds prompt formatting and response parsing through one
    # explicit protocol. Other benchmarks leave the official-compatible
    # default untouched.
    "ovobench_protocol": "official_compatible",
    "use_internal_kv": True,
    # internal_kv: keep the ShallowStream path and continue from cached lower KV.
    # simple_prompt: keep retrieval/clustering, then rebuild a SimpleStream-like
    # full prompt from selected visual embeddings and call model.generate().
    "selected_generate_mode": "simple_prompt",
    # If false, selected visual memories are inserted without Qwen3-VL textual
    # timestamp tokens such as "<3.0 seconds>".
    "selected_prompt_use_timestamps": False,
    # Re-encode every visual unit used by simple_prompt through Qwen3-VL's
    # native image path.  This includes long-cluster representatives,
    # retrieved short units, and recent units.  Disabled preserves the
    # historical behavior where only recent units are re-encoded as images.
    "selected_prompt_reencode_all_as_images": False,
    # Expand each selected recent temporal unit back to the sampled frames
    # that Qwen's temporal patching merged into it.  With all-image re-encoding
    # enabled, the same expansion is also applied to retrieved short units.
    "selected_prompt_expand_recent_source_frames": True,
    # Canonical Origin simple-prompt generation does not inject DeepStack.
    # Keep it as an explicit ablation rather than silently changing the method.
    "selected_prompt_use_deepstack": False,
    # standard preserves the upstream OVO-Bench multiple-choice instruction.
    "mcq_prompt_mode": "standard",
    # Zero-based index of the first deep layer; shallow layers are
    # layers[:prune_layer].
    "prune_layer": 8,
    "rope_position_mode": "relative",
    "retrieval_topk_units": 4,
    "retrieval_search_last_n_units": 64,
    "retrieval_expand_prev_units": 1,
    "retrieval_expand_next_units": 0,
    "retrieval_expand_prev_stride_units": 2,
    "retrieval_expand_next_stride_units": 1,
    # temporal_neighbors applies the configured expansion. score_fill keeps
    # its exact per-sample unit count but fills the extra slots by score rank.
    "retrieval_expansion_strategy": "temporal_neighbors",
    # Enforce the per-unit local window. The FlashAttention path slices local
    # KV views and concatenates the persistent sink instead of building a dense
    # causal/window mask.
    "streaming_lower_mask": True,
    # Canonical local-attention window in sampled video frames. Qwen converts
    # this to temporal units and then to the actual token span produced by the
    # current video grid; users should not tune a resolution-dependent token
    # count directly.
    "use_rekv_sink": True,
    "shallow_prefill_local_window_frames": 128,
    # Observation protocols can keep an exact raw-KV archive while bounding
    # the GPU-side cache used by subsequent streaming appends.
    "streaming_archive_full_history": False,
    "streaming_archive_device": "cpu",
    # Offline evaluation can simulate open-window streaming by pre-filling
    # sampled video frames in chunks and promoting old detail frames to long
    # clusters after each chunk. 0 restores the old full-video prefill path.
    "streaming_prefill_batch_frames": 64,
    "retrieval_score_order": "highest",
    # Unit-ranking strategy. shallow_unit_cosine preserves the historical
    # final-shallow-layer ranking. shallow_layer_token_vote lets every retained
    # shallow layer vote with its top-attended historical visual tokens.
    "retrieval_score_strategy": "shallow_unit_cosine",
    "retrieval_vote_layer_start": 0,
    "retrieval_vote_topk_tokens_per_layer": 64,
    # all_mean averages attention from every question-content token.
    # prompt_last uses the final token of the complete model input prompt.
    "retrieval_vote_query_token_mode": "all_mean",
    # Optional DivPrune-style max-min selection inside a relevance-filtered
    # frame pool. The pool is ranked by layer-token votes first.
    "retrieval_vote_diversity_mode": "off",
    "retrieval_vote_diversity_pool_multiplier": 4,
    # Read-only evaluation observation. When enabled, persist the shallow Q/K
    # cosine score for every configured recent temporal unit. The observation
    # is attached after scoring and never participates in gate or retrieval
    # decisions.
    "observe_recent_retrieval_unit_scores": False,
    # Evidence-ranking backend. ``shallow`` preserves the production Q/K
    # ranking. ``siglip`` is an opt-in query-choice-gated ablation which ranks
    # complete visual units with an auxiliary SigLIP image/text cosine score.
    "evidence_retrieval_backend": "shallow",
    "evidence_retrieval_query_source": "question_text",
    "evidence_retrieval_siglip_model_path": "",
    "evidence_retrieval_siglip_device": "auto",
    "evidence_retrieval_siglip_batch_size": 16,
    # unit keeps the original video-temporal-unit retrieval. token keeps the
    # same short-context token budget, but spends it on individual visual tokens
    # from the searchable history.
    "retrieval_selection_granularity": "unit",
    # off preserves the existing retrieval policy. latest_unit_score gates on
    # the existing shallow retrieval score for the newest temporal unit.
    # anchor_hidden compares question hidden states against fixed anchors;
    # anchor_kq remains available as a legacy diagnostic.
    "task_gate_mode": "off",
    # full_prompt preserves the historical MCQ routing input. Set question_text
    # explicitly for the raw-question-only cross-backbone ablation.
    "task_gate_input_source": "full_prompt",
    "task_gate_query_choice_prompt_version": "retrieval_ab_v1",
    "task_gate_query_choice_threshold": 0.0,
    # Full-model A/B sufficiency gate over the newest temporal units. Its
    # recent-only answer path reuses the gate prefill KV.
    "task_gate_recent_sufficiency_units": 2,
    "task_gate_recent_sufficiency_threshold": 0.0,
    "task_gate_replay_path": "",
    # Zero-based layer index used by the Anchor-KQ probe, independent from the
    # zero-based deep-layer start in prune_layer. None probes prune_layer - 1,
    # i.e. the final shallow layer.
    "task_gate_anchor_kq_layer": None,
    "task_gate_anchor_kq_threshold": -0.006516069,
    # Zero-based semantic-probe layer, independent from prune_layer. None uses
    # the final shallow layer. The score is always past cosine minus non-past
    # cosine, and retrieval always uses score >= threshold.
    "task_gate_anchor_hidden_layer": None,
    "task_gate_anchor_hidden_threshold": 0.0,
    # Attention-distribution gate: use exact matching-head scaled QK logits
    # from a retained shallow layer. Its visual-history evidence is a
    # cardinality-corrected history-vs-recent log-attention ratio. An optional
    # named linear classifier combines the semantic prior with attention
    # distribution statistics; an empty classifier preserves the original
    # scalar-fusion experiment.
    "task_gate_attention_layer": None,
    # Optional zero-based shallow layers to record alongside the selected
    # attention layer. Empty records only task_gate_attention_layer.
    "task_gate_attention_observation_layers": [],
    "task_gate_attention_weight": 0.0,
    "task_gate_attention_classifier": {},
    "task_gate_attention_threshold": 0.0,
    # Layerwise history-attention gate. The three variants compare log
    # history-attention change across every retained shallow layer.
    "task_gate_history_decay_variant": "endpoint_delta",
    "task_gate_history_decay_threshold": 0.0,
    "task_gate_past_anchor": DEFAULT_TASK_GATE_PAST_ANCHOR,
    "task_gate_nonpast_anchor": DEFAULT_TASK_GATE_NONPAST_ANCHOR,
    "task_gate_text_model_path": "",
    "allow_transductive_task_gate": False,
    "task_gate_text_threshold": 0.0,
    "task_gate_probe_strategy": "temporal_topk",
    "task_gate_probe_threshold": 0.0,
    "task_gate_probe_temperature": 0.07,
    # Latest-unit shallow retrieval score gate. Retrieval is enabled when the
    # normalized latest-unit K dot normalized question Q is <= this threshold.
    "task_gate_latest_unit_score_threshold": 0.0095,
    "task_gate_realtime_policy": "recent_only",
    # Optional recent-window override used only when the gate selects
    # recent_only. None preserves retrieval_recent_units for both policies.
    "task_gate_realtime_recent_units": None,
}

def _load_config() -> Dict[str, object]:
    return apply_config_sources(
        DEFAULT_CONFIG,
        name=MODEL_NAME,
        default_file="configs/shallowstream/qwen3vl_v3_ovobench.json",
        file_envs=("SHALLOWSTREAM_QWEN3VL_V3_CONFIG_FILE",),
        json_envs=("SHALLOWSTREAM_QWEN3VL_V3_CONFIG_JSON",),
        print_prefix=f"[{MODEL_NAME} Config Override]",
    )

def _as_int(config: Dict[str, object], key: str) -> int:
    return int(config[key])

def _as_int_alias(config: Dict[str, object], key: str, old_key: str, default: int = 0) -> int:
    if key in config:
        return int(config[key])
    if old_key in config:
        return int(config[old_key])
    return int(default)

def _as_float(config: Dict[str, object], key: str) -> float:
    return float(config[key])

def _as_bool(config: Dict[str, object], key: str) -> bool:
    value = config[key]
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _retrieval_expansion_strategy(config: Dict[str, object]) -> str:
    strategy = str(
        config.get("retrieval_expansion_strategy", "temporal_neighbors")
    ).strip().lower()
    if strategy not in {"temporal_neighbors", "score_fill"}:
        raise ValueError(
            "retrieval_expansion_strategy must be 'temporal_neighbors' or "
            f"'score_fill', got {strategy!r}"
        )
    return strategy


def _full_kv_enabled(config: Dict[str, object]) -> bool:
    value = config.get("full_kv_mode", False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

def _torch_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"auto", ""}:
        return "auto"
    raise ValueError(f"Unsupported torch_dtype: {name}")
