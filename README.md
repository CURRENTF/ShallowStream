# ShallowStream Core Method

This anonymous snapshot contains only the core ShallowStream method source for
Qwen3-VL and LLaVA-OneVision. The benchmark runners, datasets, comparison
methods, experiment scripts, tests, generated artifacts, and result records are
intentionally not included.

## Included

- `src/shallowstream/qwen3vl/`: Qwen3-VL shallow prefill, streaming memory,
  routing, retrieval, selected-context reconstruction, and decoding.
- `src/shallowstream/onevision/`: LLaVA-OneVision V3 implementation of the same
  method, including its frame-native memory and retrieval path.
- `src/shallowstream/common.py`, `evidence_retrieval.py`, `task_gate.py`, and
  `history_decay_gate.py`: model-shared method primitives.
- `src/config.py`, `src/modelclass.py`, and the small `src/utils/` subset:
  internal dependencies required by the two runtimes.
- `configs/shallowstream/`: runtime defaults and one current reference setting
  for each model family. Model checkpoint paths remain unset.
- `examples/smoke_inference.py`: one-video non-empty generation check shared by
  the two standalone runtimes.

## Scope

This branch is a method-code snapshot, not a complete evaluation release. It
does not include benchmark adapters or data and therefore does not reproduce
reported benchmark scores by itself.

The Python package declares the direct runtime dependencies. FlashAttention is
kept as an optional CUDA dependency because its installation must match the
target PyTorch and CUDA build.

Run a short generation after supplying a local checkpoint and video:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python examples/smoke_inference.py \
  --model qwen3vl \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --video /path/to/video.mp4
```

Use `--model onevision` with an LLaVA-OneVision Hugging Face checkpoint for the
other runtime.
