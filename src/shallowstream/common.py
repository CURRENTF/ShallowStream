"""Model-agnostic primitives shared by ShallowStream runtimes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

def get_active_attn_implementation(model: Any) -> str:
    """Return the active Hugging Face attention backend across common model wrappers."""

    configs = []
    if model is not None:
        language = getattr(model, "language_model", None)
        if language is not None and hasattr(language, "config"):
            configs.append(language.config)
        model_core = getattr(model, "model", None)
        if model_core is not None:
            core_language = getattr(model_core, "language_model", None)
            if core_language is not None and hasattr(core_language, "config"):
                configs.append(core_language.config)
            if hasattr(model_core, "config"):
                configs.append(model_core.config)
        if hasattr(model, "config"):
            text_config = getattr(model.config, "text_config", None)
            if text_config is not None:
                configs.append(text_config)
            configs.append(model.config)

    for config in configs:
        implementation = getattr(config, "_attn_implementation", None) or getattr(
            config, "attn_implementation", None
        )
        if implementation:
            return str(implementation)
    return ""


def expand_temporal_neighbors(
    seed_ids: Iterable[int],
    candidate_ids: Iterable[int],
    *,
    previous: int = 0,
    following: int = 0,
    previous_stride: int = 1,
    following_stride: int = 1,
) -> List[int]:
    """Expand selected temporal ids without crossing the candidate set."""

    valid_ids = {int(item_id) for item_id in candidate_ids}
    expanded = {int(item_id) for item_id in seed_ids if int(item_id) in valid_ids}
    previous = max(0, int(previous))
    following = max(0, int(following))
    previous_stride = max(1, int(previous_stride))
    following_stride = max(1, int(following_stride))

    for item_id in tuple(expanded):
        for distance in range(1, previous + 1):
            neighbor = item_id - distance * previous_stride
            if neighbor in valid_ids:
                expanded.add(neighbor)
        for distance in range(1, following + 1):
            neighbor = item_id + distance * following_stride
            if neighbor in valid_ids:
                expanded.add(neighbor)
    return sorted(expanded)


def select_temporal_retrieval_ids(
    seed_ids: Iterable[int],
    ranked_ids: Iterable[int],
    candidate_ids: Iterable[int],
    *,
    previous: int = 0,
    following: int = 0,
    previous_stride: int = 1,
    following_stride: int = 1,
    strategy: str = "temporal_neighbors",
) -> Tuple[List[int], List[int]]:
    """Select retrieval ids and return the neighbor-expanded reference ids.

    ``score_fill`` preserves the exact per-sample cardinality produced by the
    configured temporal expansion, but spends the additional slots on the
    next highest-ranked candidates instead of temporal neighbors.
    """

    seed_ids = [int(item_id) for item_id in seed_ids]
    ranked_ids = [int(item_id) for item_id in ranked_ids]
    candidate_ids = [int(item_id) for item_id in candidate_ids]
    reference_ids = expand_temporal_neighbors(
        seed_ids,
        candidate_ids,
        previous=previous,
        following=following,
        previous_stride=previous_stride,
        following_stride=following_stride,
    )
    normalized = str(strategy).strip().lower()
    if normalized == "temporal_neighbors":
        return reference_ids, reference_ids
    if normalized != "score_fill":
        raise ValueError(
            "retrieval_expansion_strategy must be 'temporal_neighbors' or "
            f"'score_fill', got {strategy!r}"
        )

    valid_ids = {int(item_id) for item_id in candidate_ids}
    selected = {
        int(item_id) for item_id in seed_ids if int(item_id) in valid_ids
    }
    target_count = len(reference_ids)
    for item_id in ranked_ids:
        item_id = int(item_id)
        if item_id in valid_ids:
            selected.add(item_id)
        if len(selected) >= target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            "score_fill could not match temporal expansion cardinality: "
            f"selected={len(selected)}, target={target_count}"
        )
    return sorted(selected), reference_ids


class LayerIndexedLegacyCache:
    """Cache-compatible storage for decoder layers indexed by layer id."""

    def __init__(self, initial: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None) -> None:
        self.storage: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = dict(initial or {})

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.storage:
            old_key, old_value = self.storage[layer_idx]
            old_key = old_key.to(device=key_states.device, dtype=key_states.dtype)
            old_value = old_value.to(device=value_states.device, dtype=value_states.dtype)
            key_states = torch.cat([old_key, key_states], dim=-2)
            value_states = torch.cat([old_value, value_states], dim=-2)
        self.storage[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx in self.storage:
            return int(self.storage[layer_idx][0].shape[-2])
        if self.storage:
            first_key = next(iter(self.storage.values()))[0]
            return int(first_key.shape[-2])
        return 0

    def get_layer_seq_length(self, layer_idx: int) -> int:
        if layer_idx in self.storage:
            return int(self.storage[layer_idx][0].shape[-2])
        return 0

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.storage[layer_idx]


class SingleLayerLegacyCache:
    """Expose one legacy KV tuple through the cache interface expected by HF layers."""

    def __init__(self, kv: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        self.key = kv[0] if kv is not None else None
        self.value = kv[1] if kv is not None else None

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = None) -> int:
        if self.key is None:
            return 0
        return int(self.key.shape[-2])

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        if self.key is None:
            return 0
        return int(self.key.shape[-2])

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: Optional[int] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            self.key = key_states
            self.value = value_states
        else:
            self.key = torch.cat([self.key, key_states], dim=-2)
            self.value = torch.cat([self.value, value_states], dim=-2)
        return self.key, self.value

    def to_legacy(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.key is None or self.value is None:
            return None
        return self.key, self.value
