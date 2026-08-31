# ShallowStream

This anonymous snapshot contains the core ShallowStream method source for Qwen3-VL and LLaVA-OneVision.

## Code Structure

- `src/shallowstream/qwen3vl/`: Qwen3-VL shallow prefill, streaming memory,
  routing, retrieval, selected-context reconstruction, and decoding.
- `src/shallowstream/onevision/`: LLaVA-OneVision shallow prefill, frame-native
  memory, routing, retrieval, selected-context reconstruction, and decoding.
- `src/shallowstream/common.py`, `evidence_retrieval.py`, `task_gate.py`, and
  `history_decay_gate.py`: shared method primitives.
- `src/config.py`, `src/modelclass.py`, and `src/utils/`: runtime configuration
  and utilities used by both model families.
- `configs/shallowstream/`: reference runtime configurations for Qwen3-VL and
  LLaVA-OneVision.
- `examples/smoke_inference.py`: standalone inference example for both models.

## Example Usage

Install the package:

```bash
pip install -e .
```

Run Qwen3-VL:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python examples/smoke_inference.py \
  --model qwen3vl \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --video /path/to/video.mp4
```

Run LLaVA-OneVision:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python examples/smoke_inference.py \
  --model onevision \
  --model-path /path/to/LLaVA-OneVision \
  --video /path/to/video.mp4
```
