# ShallowStream Implementations

This directory contains the core method implementation for the two supported
model families.

- `common.py`: shared cache, attention-backend, and temporal-selection
  primitives.
- `evidence_retrieval.py`, `task_gate.py`, and `history_decay_gate.py`: shared
  evidence selection and routing logic.
- `qwen3vl/`: Qwen3-VL configuration, temporal-unit memory, shallow model
  execution, routing, retrieval, prompt reconstruction, and decode path.
- `onevision/`: LLaVA-OneVision V3 configuration, frame-native memory, shallow
  model execution, routing, retrieval, prompt reconstruction, and decode path.

Benchmark-specific adapters, comparison methods, prototypes, standalone
observation workflows, and ablation-only model variants are outside this
snapshot.
