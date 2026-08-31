"""ShallowStream LLaVA-OneVision V3 runtime configuration and benchmark adapter."""

from __future__ import annotations

import atexit
from copy import deepcopy

from src.shallowstream.onevision.runtime import (
    LONG_CLUSTER_COSINE_SIM_THRESHOLD,
    NO_NEW_VIDEO_CHUNK,
    ShallowStreamLLaVAOneVisionBase,
    LLaVAOneVisionRuntime,
    ONEVISION_V3_DEFAULT_CONFIG,
)
from src.shallowstream.common import SingleLayerLegacyCache
from src.shallowstream.onevision.retrieval_instrumentation import (
    RetrievalObservationMixin,
)


V3_CONFIG = deepcopy(ONEVISION_V3_DEFAULT_CONFIG)


class LLaVAOneVisionV3Runtime(RetrievalObservationMixin, LLaVAOneVisionRuntime):
    version = 3
    default_config_file = "configs/shallowstream/onevision_v3.json"
    cluster_chunk_debug_dir = "./outputs/streamingbench/debug/shallowstream_v3_cluster_size"


_runtime = LLaVAOneVisionV3Runtime(V3_CONFIG)
atexit.register(_runtime._flush_retrieval_observation_summary)
CLUSTER_CHUNK_DEBUG_DIR = LLaVAOneVisionV3Runtime.cluster_chunk_debug_dir
_SingleLayerLegacyCache = SingleLayerLegacyCache
_state = _runtime.state

# Private compatibility aliases retained for existing research scripts.
_apply_v3_config_overrides_from_env = _runtime._apply_config_overrides_from_env
_ensure_model_loaded = _runtime._ensure_model_loaded
_run_stream_step = _runtime._run_stream_step
_run_single = _runtime._run_single


def __getattr__(name):
    runtime_name = {
        "_asr_model": "_asr_model",
        "_asr_unavailable_reason": "_asr_unavailable_reason",
        "_v3_env_overrides_applied": "_env_overrides_applied",
    }.get(name, name)
    if hasattr(_runtime, runtime_name):
        return getattr(_runtime, runtime_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ShallowStreamLLaVAOneVisionV3(ShallowStreamLLaVAOneVisionBase):
    runtime = _runtime
    model_name = "ShallowStream-LLaVA-OneVision-V3"
