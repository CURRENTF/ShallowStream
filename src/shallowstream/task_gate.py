"""Backbone-independent task-gate policies shared by ShallowStream runtimes."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


TASK_GATE_INPUT_SOURCES = ("question_text", "full_prompt")
QUERY_CHOICE_PROMPT_VERSIONS = (
    "retrieval_ab_v1",
    "retrieval_ab_fewshot3_v1",
    "retrieval_ab_fewshot5_v1",
    "retrieval_ab_fewshot10_v1",
)

QUERY_CHOICE_FEWSHOT10 = (
    ("What object is the person holding in the latest visible moment?", "B"),
    ("Where did the person leave the keys near the beginning of the video?", "A"),
    ("What word is currently displayed on the screen?", "B"),
    ("How many times did the bell ring over the course of the video?", "A"),
    ("Which item did the cook pick up first?", "A"),
    ("What color is the vehicle visible in the latest scene?", "B"),
    ("What was inside the box before it was emptied?", "A"),
    ("What action is the person performing right now?", "B"),
    ("Which object is immediately to the left of the cup in the latest frame?", "B"),
    ("How has the room changed compared with its appearance earlier in the video?", "A"),
)

# Keep the few-shot demonstrations frozen and deterministic across backbones.
# The 3-shot and 5-shot subsets use the requested retrieval:local label ratios
# (A:B) of 1:2 and 2:3 respectively.  The prompt version, rather than an
# implicit slice rule, is recorded in every gate decision for reproducibility.
QUERY_CHOICE_FEWSHOT3 = (
    QUERY_CHOICE_FEWSHOT10[0],  # B: local
    QUERY_CHOICE_FEWSHOT10[1],  # A: retrieval
    QUERY_CHOICE_FEWSHOT10[2],  # B: local
)
QUERY_CHOICE_FEWSHOT5 = (
    QUERY_CHOICE_FEWSHOT10[0],  # B: local
    QUERY_CHOICE_FEWSHOT10[1],  # A: retrieval
    QUERY_CHOICE_FEWSHOT10[2],  # B: local
    QUERY_CHOICE_FEWSHOT10[3],  # A: retrieval
    QUERY_CHOICE_FEWSHOT10[5],  # B: local
)

QUERY_CHOICE_FEWSHOT_DEMONSTRATIONS = {
    "retrieval_ab_fewshot3_v1": QUERY_CHOICE_FEWSHOT3,
    "retrieval_ab_fewshot5_v1": QUERY_CHOICE_FEWSHOT5,
    "retrieval_ab_fewshot10_v1": QUERY_CHOICE_FEWSHOT10,
}


def resolve_task_gate_input(
    config: Mapping[str, Any],
    *,
    question_text: str,
    full_prompt: str,
) -> Tuple[str, str]:
    """Resolve the configured gate input without changing generation input."""

    source = str(
        config.get("task_gate_input_source", "full_prompt") or "full_prompt"
    ).strip().lower()
    if source not in TASK_GATE_INPUT_SOURCES:
        raise ValueError(
            f"Unsupported task_gate_input_source={source!r}; expected one of "
            f"{list(TASK_GATE_INPUT_SOURCES)!r}"
        )
    value = question_text if source == "question_text" else full_prompt
    text = str(value or "")
    if not text.strip():
        raise ValueError(
            f"task_gate_input_source={source!r} resolved to empty text"
        )
    return text, source


def task_gate_input_metadata(text: str, source: str) -> Dict[str, Any]:
    encoded = str(text).encode("utf-8")
    return {
        "task_gate_input_source": str(source),
        "task_gate_input_sha256": hashlib.sha256(encoded).hexdigest(),
        "task_gate_input_chars": len(str(text)),
    }


def query_source_label(source: str, representation: str) -> str:
    prefix = "question_text_only" if source == "question_text" else "full_prompt"
    return f"{prefix}_{representation}"


def normalize_query_choice_prompt_version(config: Mapping[str, Any]) -> str:
    version = str(
        config.get("task_gate_query_choice_prompt_version", "retrieval_ab_v1")
        or "retrieval_ab_v1"
    ).strip()
    if version not in QUERY_CHOICE_PROMPT_VERSIONS:
        raise ValueError(
            "Unsupported task_gate_query_choice_prompt_version="
            f"{version!r}; expected one of {list(QUERY_CHOICE_PROMPT_VERSIONS)!r}"
        )
    return version


def build_query_choice_router_prompt(prompt: str, version: str) -> str:
    if version in QUERY_CHOICE_FEWSHOT_DEMONSTRATIONS:
        demonstrations = [
            f"Example {index}:\n"
            f"Video question: {question}\n"
            f"Correct routing option: {label}"
            for index, (question, label) in enumerate(
                QUERY_CHOICE_FEWSHOT_DEMONSTRATIONS[version], start=1
            )
        ]
        return (
            "You are deciding whether answering a video question requires retrieving older "
            "video memory. The system already has the latest video segment immediately "
            "available. Do not answer the video question itself.\n\n"
            "Routing options:\n"
            "A. Answering reliably requires inspecting earlier video memory.\n"
            "B. The latest video segment is sufficient; older video memory is unnecessary.\n\n"
            "Here are labeled routing examples:\n\n"
            + "\n\n".join(demonstrations)
            + "\n\nNow route this video question:\n"
            "<video_question>\n"
            f"{prompt}\n"
            "</video_question>\n\n"
            "Output only A or B.\n"
            "Routing answer:"
        )
    if version != "retrieval_ab_v1":
        raise ValueError(
            f"Unsupported query-choice router prompt version: {version!r}"
        )
    return (
        "You are deciding whether answering a video question requires retrieving older "
        "video memory. The system already has the latest video segment immediately "
        "available. Do not answer the video question itself.\n\n"
        "<video_question>\n"
        f"{prompt}\n"
        "</video_question>\n\n"
        "Choose exactly one routing option:\n"
        "A. Answering reliably requires inspecting earlier video memory.\n"
        "B. The latest video segment is sufficient; older video memory is unnecessary.\n\n"
        "Output only A or B.\n"
        "Routing answer:"
    )


def build_recent_context_sufficiency_prompt(question: str) -> str:
    """Append a removable sufficiency check after an answer-ready question."""

    question = str(question or "").strip()
    if not question:
        raise ValueError("recent-context sufficiency question must not be empty")
    return (
        f"{question}\n\n"
        "Using only the recent video evidence shown above, decide whether the "
        "question can be answered reliably. Do not answer the question yet.\n\n"
        "A. Earlier video evidence must be retrieved before answering.\n"
        "B. The recent video evidence is sufficient to answer.\n\n"
        "Output only A or B.\n"
        "Decision:"
    )


def resolve_query_choice_token_ids(tokenizer: Any) -> Tuple[int, int]:
    resolved = []
    for label in ("A", "B"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                "query_choice_logits requires each route label to be exactly one token; "
                f"{label!r} encoded as {token_ids!r}"
            )
        resolved.append(int(token_ids[0]))
    if resolved[0] == resolved[1]:
        raise RuntimeError(
            "query_choice_logits route labels A and B map to the same token id"
        )
    return resolved[0], resolved[1]


def query_choice_token_strings(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> Tuple[str, str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
        if isinstance(tokens, (list, tuple)) and len(tokens) == 2:
            return str(tokens[0]), str(tokens[1])
    return "A", "B"


def build_query_choice_decision(
    *,
    retrieve_logit: float,
    recent_logit: float,
    retrieve_token_id: int,
    recent_token_id: int,
    choice_tokens: Sequence[str],
    router_prompt_version: str,
    source: str,
    input_text: str,
    input_source: str,
    recent_policy: str = "recent_only",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    if int(retrieve_token_id) == int(recent_token_id):
        raise ValueError("query_choice_logits route token ids must be distinct")
    retrieve_logit = float(retrieve_logit)
    recent_logit = float(recent_logit)
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("query-choice threshold must be finite")
    maximum = max(retrieve_logit, recent_logit)
    retrieve_exp = math.exp(retrieve_logit - maximum)
    recent_exp = math.exp(recent_logit - maximum)
    denominator = retrieve_exp + recent_exp
    retrieval_probability = retrieve_exp / denominator
    recent_probability = recent_exp / denominator
    score = retrieve_logit - recent_logit
    retrieval_enabled = score >= threshold
    decision = {
        "enabled": True,
        "mode": "query_choice_logits",
        "source": str(source),
        "predicted_task_type": (
            "retrieval_required" if retrieval_enabled else "recent_sufficient"
        ),
        "selected_policy": "retrieval" if retrieval_enabled else str(recent_policy),
        "retrieval_enabled": bool(retrieval_enabled),
        "score": float(score),
        "threshold": threshold,
        "router_output": "single_prefill_next_token_logits",
        "router_prompt_version": str(router_prompt_version),
        "gate_model_forward_count": 1,
        "choice_logits": {
            "retrieval": retrieve_logit,
            "recent_only": recent_logit,
        },
        "choice_probabilities": {
            "retrieval": retrieval_probability,
            "recent_only": recent_probability,
        },
        "choice_token_ids": {
            "retrieval": int(retrieve_token_id),
            "recent_only": int(recent_token_id),
        },
        "choice_tokens": {
            "retrieval": str(choice_tokens[0]),
            "recent_only": str(choice_tokens[1]),
        },
        "query_source": query_source_label(input_source, "query_choice_logits"),
    }
    decision.update(task_gate_input_metadata(input_text, input_source))
    return decision


def build_anchor_hidden_decision(
    *,
    past_cosine: float,
    nonpast_cosine: float,
    threshold: float,
    probe_layer: int,
    source: str,
    input_text: str,
    input_source: str,
    recent_policy: str = "recent_only",
) -> Dict[str, Any]:
    retrieval_score = float(past_cosine) - float(nonpast_cosine)
    retrieval_enabled = retrieval_score >= float(threshold)
    decision = {
        "enabled": True,
        "mode": "anchor_hidden",
        "source": str(source),
        "predicted_task_type": "backward" if retrieval_enabled else "realtime",
        "selected_policy": "retrieval" if retrieval_enabled else str(recent_policy),
        "score": retrieval_score,
        "retrieval_score": retrieval_score,
        "threshold": float(threshold),
        "probe_layer": int(probe_layer),
        "past_cosine": float(past_cosine),
        "nonpast_cosine": float(nonpast_cosine),
        "retrieval_enabled": bool(retrieval_enabled),
        "query_source": query_source_label(input_source, "content_mean_hidden"),
        "pool": "raw_text_content_mean",
        "representation": "post_layer_hidden_state",
        "rule": "enable_retrieval_if_past_minus_nonpast_hidden_cosine_gte_threshold",
    }
    decision.update(task_gate_input_metadata(input_text, input_source))
    return decision
