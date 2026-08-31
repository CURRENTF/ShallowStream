"""Shared evidence-ranking backends for ShallowStream ablations."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image


EVIDENCE_RETRIEVAL_BACKENDS = ("shallow", "siglip")
EVIDENCE_RETRIEVAL_QUERY_SOURCES = ("full_prompt", "question_text")


def evidence_retrieval_backend(config: Mapping[str, Any]) -> str:
    backend = str(config.get("evidence_retrieval_backend", "shallow") or "shallow").strip().lower()
    if backend not in EVIDENCE_RETRIEVAL_BACKENDS:
        raise ValueError(
            f"evidence_retrieval_backend must be one of {EVIDENCE_RETRIEVAL_BACKENDS}, "
            f"got {backend!r}"
        )
    return backend


def validate_evidence_retrieval_config(
    config: Mapping[str, Any],
    *,
    model_family: str,
) -> str:
    """Validate the opt-in SigLIP ablation without changing shallow defaults."""

    backend = evidence_retrieval_backend(config)
    query_source = str(
        config.get("evidence_retrieval_query_source", "question_text")
        or "question_text"
    ).strip().lower()
    if query_source not in EVIDENCE_RETRIEVAL_QUERY_SOURCES:
        raise ValueError(
            "evidence_retrieval_query_source must be one of "
            f"{EVIDENCE_RETRIEVAL_QUERY_SOURCES}, got {query_source!r}"
        )
    batch_size = int(config.get("evidence_retrieval_siglip_batch_size", 16))
    if batch_size <= 0:
        raise ValueError("evidence_retrieval_siglip_batch_size must be positive")
    device = str(config.get("evidence_retrieval_siglip_device", "auto") or "auto").strip()
    if not device:
        raise ValueError("evidence_retrieval_siglip_device must be non-empty")
    if device != "auto":
        try:
            torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid evidence_retrieval_siglip_device={device!r}"
            ) from exc
    if backend == "shallow":
        return backend

    model_path = str(config.get("evidence_retrieval_siglip_model_path", "") or "").strip()
    if not model_path:
        raise ValueError(
            "evidence_retrieval_siglip_model_path is required when "
            "evidence_retrieval_backend='siglip'"
        )
    if str(config.get("task_gate_mode", "off") or "off").strip().lower() != "query_choice_logits":
        raise ValueError(
            "SigLIP evidence retrieval is defined only as a query_choice_logits-gated "
            "ablation; task_gate_mode must be 'query_choice_logits'"
        )
    if str(config.get("retrieval_score_order", "highest") or "highest").strip().lower() != "highest":
        raise ValueError("SigLIP cosine similarity requires retrieval_score_order='highest'")
    if int(config.get("long_cluster_topk", 0) or 0) != 0:
        raise ValueError(
            "SigLIP evidence retrieval cannot rank KV-only long clusters; "
            "long_cluster_topk must be 0"
        )
    family = str(model_family).strip().lower()
    if family == "qwen3vl":
        if bool(config.get("long_cluster_enabled", False)):
            raise ValueError(
                "Qwen3-VL SigLIP evidence retrieval requires long_cluster_enabled=false"
            )
        granularity = str(
            config.get("retrieval_selection_granularity", "unit") or "unit"
        ).strip().lower()
        if granularity != "unit":
            raise ValueError(
                "SigLIP ranks complete visual units; "
                "retrieval_selection_granularity must be 'unit'"
            )
        if int(config.get("retrieval_topk_units", 0) or 0) <= 0:
            raise ValueError("Qwen3-VL SigLIP evidence retrieval requires retrieval_topk_units > 0")
    elif family == "onevision":
        if float(config.get("retrieval_subtitle_weight", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "SigLIP evidence retrieval is direct image-text similarity; "
                "retrieval_subtitle_weight must be 0"
            )
        if int(config.get("retrieval_topk_frames", 0) or 0) <= 0:
            raise ValueError("OneVision SigLIP evidence retrieval requires retrieval_topk_frames > 0")
    else:
        raise ValueError(f"Unsupported evidence retrieval model family: {model_family!r}")
    return backend


def resolve_evidence_query(
    config: Mapping[str, Any],
    *,
    full_prompt: str,
    question_text: str,
) -> tuple[str, str]:
    source = str(
        config.get("evidence_retrieval_query_source", "question_text")
        or "question_text"
    ).strip().lower()
    if source not in EVIDENCE_RETRIEVAL_QUERY_SOURCES:
        raise ValueError(f"Unsupported evidence_retrieval_query_source={source!r}")
    query = str(question_text if source == "question_text" else full_prompt).strip()
    if not query:
        raise ValueError(
            f"SigLIP evidence retrieval received an empty {source} query; "
            "no full-prompt fallback is applied"
        )
    return query, source


@dataclass(frozen=True)
class SigLIPScoreResult:
    scores: tuple[float, ...]
    metadata: Dict[str, Any]


class SigLIPEvidenceRetriever:
    """Lazy auxiliary SigLIP scorer using the repository's Transformers dependency."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from transformers import AutoModel, AutoProcessor

        configured_path = str(config["evidence_retrieval_siglip_model_path"]).strip()
        configured_device = str(
            config.get("evidence_retrieval_siglip_device", "auto") or "auto"
        ).strip()
        self.batch_size = int(config.get("evidence_retrieval_siglip_batch_size", 16))
        self.model_path = (
            os.path.realpath(configured_path) if os.path.exists(configured_path) else configured_path
        )
        self.device = torch.device(
            "cuda" if configured_device == "auto" and torch.cuda.is_available()
            else "cpu" if configured_device == "auto"
            else configured_device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"SigLIP evidence retrieval requested {self.device}, but CUDA is unavailable"
            )
        self.processor = AutoProcessor.from_pretrained(configured_path)
        self.model = AutoModel.from_pretrained(configured_path)
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", ""))
        if model_type not in {"siglip", "siglip2"}:
            raise TypeError(
                "Configured evidence model must be SigLIP/SigLIP2; "
                f"resolved model_type={model_type!r}"
            )
        if not callable(getattr(self.model, "get_image_features", None)) or not callable(
            getattr(self.model, "get_text_features", None)
        ):
            raise TypeError(
                "Configured evidence model must expose SigLIP-compatible "
                "get_image_features() and get_text_features()"
            )
        self.model.to(self.device)
        self.model.eval()
        try:
            self.dtype = str(next(self.model.parameters()).dtype)
        except StopIteration:
            self.dtype = "unknown"

    @staticmethod
    def _normalize(features: torch.Tensor, *, kind: str) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError(f"SigLIP {kind} features must be a rank-2 tensor")
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def score(self, query: str, images: Sequence[Image.Image]) -> SigLIPScoreResult:
        query = str(query).strip()
        if not query:
            raise ValueError("SigLIP evidence query must be non-empty")
        normalized_images = [image.convert("RGB") for image in images]
        if not normalized_images:
            return SigLIPScoreResult(scores=(), metadata=self.metadata(query, 0))

        text_inputs = self.processor(text=[query], padding=True, return_tensors="pt")
        text_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in text_inputs.items()
        }
        with torch.inference_mode():
            text_features = self._normalize(
                self.model.get_text_features(**text_inputs),
                kind="text",
            )
            score_parts = []
            for start in range(0, len(normalized_images), self.batch_size):
                batch = normalized_images[start : start + self.batch_size]
                image_inputs = self.processor(images=batch, return_tensors="pt")
                image_inputs = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in image_inputs.items()
                }
                image_features = self._normalize(
                    self.model.get_image_features(**image_inputs),
                    kind="image",
                )
                if image_features.shape[1] != text_features.shape[1]:
                    raise RuntimeError(
                        "SigLIP image/text feature dimensions differ: "
                        f"{image_features.shape[1]} != {text_features.shape[1]}"
                    )
                score_parts.append((image_features @ text_features[0]).detach().cpu())
        scores = torch.cat(score_parts).float().tolist()
        if len(scores) != len(normalized_images):
            raise RuntimeError(
                f"SigLIP returned {len(scores)} scores for {len(normalized_images)} images"
            )
        return SigLIPScoreResult(
            scores=tuple(float(value) for value in scores),
            metadata=self.metadata(query, len(normalized_images)),
        )

    def metadata(self, query: str, candidate_count: int) -> Dict[str, Any]:
        return {
            "backend": "siglip",
            "model_path": self.model_path,
            "model_class": type(self.model).__name__,
            "model_type": str(getattr(self.model.config, "model_type", "")),
            "processor_class": type(self.processor).__name__,
            "device": str(self.device),
            "dtype": self.dtype,
            "batch_size": int(self.batch_size),
            "score": "normalized_image_text_cosine",
            "candidate_count": int(candidate_count),
            "query_sha256": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
        }


def get_siglip_evidence_retriever(owner: Any) -> SigLIPEvidenceRetriever:
    retriever = getattr(owner, "_siglip_evidence_retriever", None)
    if retriever is None:
        retriever = SigLIPEvidenceRetriever(owner.config)
        owner._siglip_evidence_retriever = retriever
    return retriever
