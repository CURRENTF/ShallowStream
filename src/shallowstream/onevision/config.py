from __future__ import annotations

from typing import Any, Dict


NO_NEW_VIDEO_CHUNK = "__NO_NEW_VIDEO_CHUNK__"
LONG_CLUSTER_COSINE_SIM_THRESHOLD = 0.99
ONEVISION_V3_DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": "",
    "device_map": "auto",
    # Use FlashAttention2 for the HF decoder layers. Lower ShallowStream/ReKV-style
    # retrieval layers are still computed by the custom attention path below.
    "attn_implementation": "flash_attention_2",
    # Keep this False so a missing flash-attn install fails loudly instead of
    # silently running the slower/default attention implementation.
    "flash_attention_fallback": False,
    # Baseline-only mode: every streamed token is evaluated by every language
    # layer and the complete historical KV is retained for answering. Runtime
    # adapters must verify the resolved layer depth before reporting FullKV.
    "full_kv_mode": False,
    # StreamingBench asks multiple questions from one growing stream. When
    # enabled, question prefill keeps an exact CPU-resident copy of the visual
    # KV instead of consuming the one-shot OVO cache.
    "fullkv_preserve_stream_history": False,
    # simple_prompt: use shallow propagation only to retrieve memory, then
    # rebuild the selected multimodal prompt and re-prefill all decoder layers.
    # internal_kv: continue from selected shallow KV/cutoff hidden states.
    "selected_generate_mode": "simple_prompt",
    # Re-encode every visual unit selected for simple_prompt through
    # OneVision's native AnyRes image path.  Long clusters contribute one
    # representative source image; retrieved and recent frames use their
    # retained RGB inputs.  The streaming prefill remains on the video path.
    "selected_prompt_reencode_all_as_images": False,
    # Re-encode only the recent simple-prompt region through OneVision's native
    # AnyRes image path. Long clusters and retrieved short-memory frames keep
    # their cached video-path embeddings. Mutually exclusive with all-images.
    "selected_prompt_reencode_recent_as_images": False,
    "sample_fps": 0.5,
    "max_frames_num": None,
    # Optional SimpleStream input policy. When set, retain only the latest N
    # sampled frames before visual preprocessing. None preserves the ordinary
    # FullKV/ShallowStream input semantics.
    "input_recent_frames": None,
    # Canonical ReKV local window in sampled frames. OneVision emits a fixed
    # n_frame_tokens visual tokens per frame, so the attention kernel derives
    # its token window deterministically from these two values.
    "shallow_prefill_local_window_frames": 128,
    "n_frame_tokens": 196,
    # ShallowStream V3: only prefill/cache lower layers [0, prune_layer)
    "prune_layer": 8,
    # Always keep the most recent N frames as short-term context.
    "retrieval_recent_frames": 6,
    # Restrict retrieval candidate pool to the latest X frames.
    # <=0 means searching across all cached frames.
    "retrieval_search_last_n_frames": 64,
    # Independent retrieval count (top-k frames). If <= 0, do not add important-frame branch.
    "retrieval_topk_frames": 4,
    # Expand important frames with temporal neighbors.
    "retrieval_expand_prev_frames": 1,
    "retrieval_expand_next_frames": 0,
    # Expansion stride for prev/next neighbors.
    # stride=2 means selecting fid±2 (skip one frame in between).
    "retrieval_expand_prev_stride": 2,
    "retrieval_expand_next_stride": 1,
    "retrieval_temperature": 1.0,
    # Rank historical frames either with the legacy final-layer averaged Q/K
    # cosine or by voting for each shallow layer's top-attended visual tokens.
    "retrieval_score_strategy": "final_layer_cosine",
    "retrieval_vote_layer_start": 0,
    "retrieval_vote_topk_tokens_per_layer": 64,
    "retrieval_vote_query_token_mode": "all_mean",
    "retrieval_vote_diversity_mode": "off",
    "retrieval_vote_diversity_pool_multiplier": 4,
    # Long-term cluster retrieval top-k (cluster-center level).
    "long_cluster_topk": 4,
    # Write per-chunk cluster size jsonl debug to CLUSTER_CHUNK_DEBUG_DIR.
    "long_cluster_debug": False,
    # Retrieval score fusion: visual similarity + subtitle similarity.
    # Final score uses weighted average on frames that have subtitle tokens.
    # Frames without subtitle tokens fall back to visual-only similarity.
    "retrieval_visual_weight": 1.0,
    "retrieval_subtitle_weight": 0.0,
    # "highest" selects the most similar frames; "lowest" selects the least similar frames.
    "retrieval_score_order": "highest",
    # Opt-in evidence-ranking ablation shared with Qwen3-VL. The default keeps
    # the original shallow-layer retrieval path bit-for-bit unchanged.
    "evidence_retrieval_backend": "shallow",
    "evidence_retrieval_query_source": "question_text",
    "evidence_retrieval_siglip_model_path": "",
    "evidence_retrieval_siglip_device": "auto",
    "evidence_retrieval_siglip_batch_size": 16,
    # Optional text-only router used by the OVO-Bench Q1/H1 adaptations.
    "task_gate_mode": "off",
    # full_prompt preserves the historical MCQ routing input. Set question_text
    # explicitly for the raw-question-only cross-backbone ablation.
    "task_gate_input_source": "full_prompt",
    "task_gate_realtime_recent_frames": 2,
    # Reuse the prune-layer retrieval Q/K vectors: retrieve history when the
    # latest sampled-frame cosine score is at or below this threshold.
    "task_gate_latest_unit_score_threshold": 0.0095,
    # Fixed zero-threshold layerwise history-attention rule. The variant
    # determines only how the sign of log attention change is summarized.
    "task_gate_history_decay_variant": "endpoint_delta",
    "task_gate_history_decay_threshold": 0.0,
    "task_gate_realtime_policy": "recent_only",
    # Observation-only, one-based shallow layer numbers. Their question-Q to
    # latest sampled-frame-K cosine scores are recorded without changing the
    # gate decision.
    "observe_latest_unit_score_layers": [],
    # Add textual section labels around selected visual embeddings when the
    # final answer uses simple_prompt reconstruction.
    "selected_prompt_use_labels": False,
    # Re-encode only the recent frames with OneVision's native AnyRes image
    # path when rebuilding the final simple prompt. Retrieved short frames and
    # long clusters continue to use their cached video embeddings.
    "selected_prompt_reencode_recent_as_images": False,
    # Versioned query-choice prompt. The few-shot version reuses the frozen
    # balanced synthetic demonstrations from the Qwen3-VL query-only study.
    "task_gate_query_choice_prompt_version": "retrieval_ab_v1",
    "task_gate_query_choice_threshold": 0.0,
    # OVO answer-prompt selection. StreamingBench does not consume this key.
    # Keep the runtime-wide fallback upstream-compatible; ShallowStream OVO
    # configs opt into the historical direct-answer prompt explicitly.
    "mcq_prompt_mode": "standard",
    "task_gate_anchor_hidden_layer": None,
    "task_gate_anchor_hidden_threshold": 0.0,
    "task_gate_past_anchor": (
        "This question is about something that happened earlier in the video. "
        "It asks the model to recall a past action, past object state, past "
        "interaction, or something that was seen before but may not be visible "
        "now. It may also require remembering repeated events across time, "
        "accumulating history, or recognizing a counting question such as how "
        "many times something happened. The question is usually phrased in the "
        "past tense or requires retrieval from earlier visual memory."
    ),
    "task_gate_nonpast_anchor": (
        "This question is about what is happening now or what is visible in the "
        "current scene. It asks about the current action, current object, current "
        "attribute, current text, or present spatial relation. The question is "
        "usually phrased in the present tense or present progressive and can "
        "often be answered from the current visible frames."
    ),
    # Optional retrieval instrumentation. This does not change selection or
    # generation semantics and is disabled for standard V3 runs.
    "observation_enabled": False,
    "observation_print": False,
    "observation_save": True,
    "observation_dir": "./outputs/streamingbench/observations/shallowstream_v3",
    "observation_recent_probe_frames": 4,
    "observation_top_m": 2,
    # Generation
    "max_new_tokens": 32,
    # Benchmark-only fixed-length decode mode. When enabled, EOS tokens do not
    # stop generation before max_new_tokens so decode latency is comparable.
    "force_exact_new_tokens": False,
    # Benchmark-only timing guard. When enabled, synchronize CUDA at the
    # prefill/query boundary and after generation so wall-clock query timing is
    # directly comparable with other implementations.
    "latency_sync_cuda": False,
    "temperature": 0.0,
    "assistant_suffix": "<|im_end|><|im_start|>assistant\n",
    # Audio transcript side-channel. Audio is first converted to timestamped
    # text, then subtitles aligned with retrieved frames are inserted into the
    # language prompt. The VLM weights and visual path stay unchanged.
    "enable_audio_transcript": False,
    # "video": transcribe the input video clip audio track.
    # "audio": transcribe the audio path passed by StreamingBench.
    # "audio_or_video": prefer the audio path, fallback to the video clip.
    "audio_source": "video",
    "asr_model_name": "base",
    "asr_device": "cuda",
    "asr_compute_type": "float16",
    "asr_beam_size": 5,
    "asr_vad_filter": True,
    "asr_cache_dir": "./outputs/streamingbench/cache/asr",
    "audio_frame_context_radius_s": 1.25,
    "audio_nearest_segment_radius_s": 2.5,
    "audio_caption_max_tokens_per_frame": 64,
    "max_audio_context_chars": 1200,
    "audio_context_prefix": (
        "Audio transcript snippets aligned with the retrieved video frames. "
        "Use them only as supporting context when they are relevant."
    ),
    # Vision batching
    "vision_batch_size": 32,
    # Micro-batch size for interleaved (visual+caption) lower-layer prefill.
    "prefill_interleave_batch_size": 32,
    "debug": False,
    "debug_mem": False,
    "debug_frames": False,
    "debug_grid_max_side": 320,
    "debug_similarity": False,
    "debug_similarity_dir": "./outputs/streamingbench/debug/retrieval",
}


def new_runtime_state() -> Dict[str, Any]:
    return {
        "session": None,
        "sample_fps": None,
        "max_frames_num": None,
        "lower_kv": None,            # Dict[int, {"k": raw_k, "v": raw_v, "pos": absolute_positions}]
        "init_len": 0,               # number of init prompt tokens
        "init_input_embeds": None,   # input embeddings retained for simple_prompt reconstruction
        "init_hidden_l8": None,      # (init_len, hidden_dim)
        "frame_input_embeds": None,  # selected visual inputs retained on CPU for simple_prompt
        "frame_source_images": [],   # RGB PIL images retained for selected-prompt re-encoding
        "frame_hidden_l8": None,     # (num_frames_window, n_frame_tokens, hidden_dim)
        "frame_source_ids": None,    # List[float], absolute sampled frame timestamps (seconds)
        "frame_spans": [],           # List[{visual_start/end, caption_start/end}] in lower_kv token positions
        "frame_caption_hidden_l8": [],  # List[(caption_len, hidden_dim)] after prune_layer
        "frame_debug_thumbs": [],    # List[np.ndarray], small RGB thumbnails aligned with frame ids
        "frame_evidence_images": [], # RGB PIL images retained only for SigLIP evidence ranking
        "audio_segments": [],        # timestamped ASR segments: [{"start", "end", "text"}]
        "frame_captions": [],        # List[str], aligned with frame_source_ids
        "question_counter": 0,
        "last_selection": {},
        "last_gate_decision": {},
        "last_latest_unit_score_observation": {},
        "last_history_decay_query": {},
        "last_history_decay_observation": {},
        "last_retrieval_query_q": {},
        "last_simple_prompt_stats": {},
        "long_clusters": [],  # List[Dict], sequential long-term clusters outside short retrieval window
        "lower_attn_path_stats": {},
        "upper_flash_layer_calls": 0,
        "upper_nonflash_layer_calls": 0,
    }
