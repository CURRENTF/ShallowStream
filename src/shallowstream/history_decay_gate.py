"""Shared zero-threshold scoring for layerwise history-attention gates."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping


HISTORY_DECAY_VARIANTS = (
    "endpoint_delta",
    "linear_slope",
    "median_step",
)


def score_history_attention_layers(
    layer_masses: Mapping[int, float],
    variant: str,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Score log history-attention change across shallow layers."""

    normalized_variant = str(variant or "").strip().lower()
    if normalized_variant not in HISTORY_DECAY_VARIANTS:
        raise ValueError(
            f"Unsupported task_gate_history_decay_variant={variant!r}; expected one of "
            f"{list(HISTORY_DECAY_VARIANTS)!r}"
        )
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("history-layer decay threshold must be finite")
    ordered = sorted((int(layer), float(mass)) for layer, mass in layer_masses.items())
    if len(ordered) < 2:
        raise ValueError("history-layer decay gate requires at least two shallow layers")
    if any(not math.isfinite(mass) or mass < 0.0 for _layer, mass in ordered):
        raise ValueError("history-layer attention masses must be finite and non-negative")

    epsilon = 1e-12
    layers = [layer for layer, _mass in ordered]
    masses = [mass for _layer, mass in ordered]
    log_masses = [math.log(max(mass, epsilon)) for mass in masses]
    steps = [right - left for left, right in zip(log_masses, log_masses[1:])]

    if normalized_variant == "endpoint_delta":
        score = log_masses[-1] - log_masses[0]
    elif normalized_variant == "linear_slope":
        x_mean = statistics.fmean(layers)
        y_mean = statistics.fmean(log_masses)
        denominator = sum((layer - x_mean) ** 2 for layer in layers)
        if denominator <= 0.0:
            raise ValueError("history-layer slope requires distinct layer indices")
        score = sum(
            (layer - x_mean) * (value - y_mean)
            for layer, value in zip(layers, log_masses)
        ) / denominator
    else:
        score = statistics.median(steps)

    return {
        "variant": normalized_variant,
        "score": float(score),
        "threshold": threshold,
        "layer_indices": layers,
        "history_attention_masses": masses,
        "log_history_attention_masses": log_masses,
        "adjacent_log_mass_steps": steps,
    }
