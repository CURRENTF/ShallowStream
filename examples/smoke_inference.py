"""Run one short, non-empty generation through either released runtime."""

from __future__ import annotations

import argparse
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen3vl", "onevision"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe the main visible action in one short sentence.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def run_qwen(args):
    from src.shallowstream.qwen3vl import ShallowStreamQwen3VLV3

    model = ShallowStreamQwen3VLV3(
        {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "max_sampled_frames": 8,
            "streaming_prefill_batch_frames": 0,
            "task_gate_mode": "off",
            "selected_generate_mode": "simple_prompt",
            "retrieval_recent_units": 2,
            "retrieval_topk_units": 2,
            "long_cluster_enabled": False,
            "long_cluster_topk": 0,
        }
    )
    return model.Run(args.video, args.prompt)


def run_onevision(args):
    override = {
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "sample_fps": 1.0,
        "max_frames_num": 8,
        "task_gate_mode": "off",
        "selected_generate_mode": "simple_prompt",
        "prune_layer": 4,
        "retrieval_recent_frames": 2,
        "retrieval_topk_frames": 2,
        "long_cluster_topk": 0,
    }
    os.environ["SHALLOWSTREAM_V3_CONFIG_OVERRIDE_JSON"] = json.dumps(override)
    from src.shallowstream.onevision import ShallowStreamLLaVAOneVisionV3

    model = ShallowStreamLLaVAOneVisionV3()
    return model.Run(args.video, args.prompt)


def main():
    args = parse_args()
    if not os.path.isfile(args.video) or os.path.getsize(args.video) <= 0:
        raise ValueError(f"Video is missing or empty: {args.video}")
    started = time.perf_counter()
    response = run_qwen(args) if args.model == "qwen3vl" else run_onevision(args)
    response = str(response).strip()
    if not response:
        raise RuntimeError(f"{args.model} returned an empty response")
    print(
        json.dumps(
            {
                "model": args.model,
                "response": response,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
