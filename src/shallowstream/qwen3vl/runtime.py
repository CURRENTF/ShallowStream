"""Standalone owner for the ShallowStream Qwen3-VL V3 engine."""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import (
    MODEL_NAME,
    _get_active_attn_implementation,
    _load_config,
    _torch_dtype,
)
from .engine import Qwen3VLInternalKVEngine
from .frame_memory import Qwen3VLFrameMemory


class ShallowStreamQwen3VLV3:
    """Load Qwen3-VL and expose the core ShallowStream inference path."""

    model_name = MODEL_NAME

    def __init__(self, config_override: Optional[Mapping[str, object]] = None) -> None:
        self.config = _load_config()
        if config_override:
            unknown = sorted(set(config_override).difference(self.config))
            if unknown:
                raise ValueError(f"Unknown Qwen3-VL config keys: {unknown}")
            self.config.update(dict(config_override))

        self.last_gate_decision = None
        self._task_gate_query_text = ""
        self._task_gate_sample_id = ""
        self.processor = self._load_processor()
        self.model = self._load_model()
        self.memory = Qwen3VLFrameMemory(self.config)
        self._initialize_runtime_tokens()
        self.kv_engine = Qwen3VLInternalKVEngine(self)

    def _load_processor(self):
        model_path = str(self.config.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("ShallowStream Qwen3-VL V3 requires model_path")
        kwargs = {}
        if os.environ.get("MIN_PIXELS"):
            kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        return AutoProcessor.from_pretrained(model_path, **kwargs)

    def _load_model(self):
        model_path = str(self.config["model_path"])
        kwargs = {
            "device_map": self.config["device_map"],
            "torch_dtype": _torch_dtype(str(self.config["torch_dtype"])),
        }
        requested_attention = str(self.config.get("attn_implementation", "")).strip()
        if requested_attention:
            kwargs["attn_implementation"] = requested_attention

        saved_world_size = os.environ.pop("WORLD_SIZE", None)
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        finally:
            if saved_world_size is not None:
                os.environ["WORLD_SIZE"] = saved_world_size

        active_attention = _get_active_attn_implementation(model)
        if requested_attention == "flash_attention_2" and active_attention != requested_attention:
            raise RuntimeError(
                "Qwen3-VL requires flash_attention_2, "
                f"but the loaded model uses {active_attention!r}"
            )
        model.eval()
        return model

    def _initialize_runtime_tokens(self) -> None:
        self._hf_model = self.model
        self._visual = getattr(getattr(self.model, "model", None), "visual", None)
        self._text_model = getattr(self.model, "model", self.model)
        self.image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        self.video_token_id = int(getattr(self.model.config, "video_token_id", 151656))

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen3-VL processor does not expose a tokenizer")
        token_names = {
            "video_token_id": "<|video_pad|>",
            "vision_start_id": "<|vision_start|>",
            "vision_end_id": "<|vision_end|>",
            "im_start_id": "<|im_start|>",
            "im_end_id": "<|im_end|>",
        }
        for attribute, token in token_names.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or int(token_id) < 0:
                raise RuntimeError(f"Qwen3-VL tokenizer is missing {token}")
            setattr(self, attribute, int(token_id))

        if self._visual is not None and hasattr(self._visual, "spatial_merge_size"):
            self.merge_size = int(self._visual.spatial_merge_size)
        else:
            image_processor = getattr(self.processor, "image_processor", None)
            self.merge_size = int(getattr(image_processor, "merge_size", 2) or 2)

        video_processor = getattr(self.processor, "video_processor", None)
        image_processor = getattr(self.processor, "image_processor", None)
        patch_size = getattr(video_processor, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(image_processor, "patch_size", None)
        if patch_size is not None:
            self.config["video_image_patch_size"] = int(patch_size)
        temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
        if temporal_patch_size is not None:
            self.config["video_temporal_patch_size"] = int(temporal_patch_size)

    def _input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def assistant_answer_prefix() -> str:
        return ""

    def run(self, video_path: str, prompt: str) -> str:
        self._task_gate_query_text = str(prompt).strip()
        response = self.kv_engine.inference(str(video_path), str(prompt))
        self.last_gate_decision = self.kv_engine.last_gate_decision
        if response is None or not str(response).strip():
            raise RuntimeError("ShallowStream Qwen3-VL returned an empty response")
        return str(response).strip()

    def Run(self, file: str, inp: str) -> str:
        return self.run(file, inp)

    def name(self) -> str:
        return self.model_name
