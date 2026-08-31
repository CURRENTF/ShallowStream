"""Optional retrieval instrumentation mixed into ShallowStream OneVision V3."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional

import torch


class RetrievalObservationMixin:
    """Collect retrieval diagnostics without changing frame selection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._observation_accumulator: Dict[str, Any] = {
            "count": 0,
            "sums": {},
            "mins": {},
            "maxs": {},
        }
        self._observation_summary_flushed = False

    def _mean_top_values(self, values: torch.Tensor, k: int) -> float:
        if not isinstance(values, torch.Tensor) or values.numel() <= 0 or k <= 0:
            return float("nan")
        kk = min(int(k), int(values.numel()))
        top_values = torch.topk(values.float(), k=kk, largest=True).values
        return float(top_values.mean().detach().cpu().item())

    def _compute_retrieval_observation_metrics(
        self,
        scores: torch.Tensor,
        candidate_idx_t: torch.Tensor,
        recent_probe_n: int,
        score_order: str,
    ) -> Dict[str, Any]:
        """Observation-only metrics for deciding whether recent frames should grow."""
        if not isinstance(scores, torch.Tensor) or scores.numel() <= 0:
            return {}
        if not isinstance(candidate_idx_t, torch.Tensor) or candidate_idx_t.numel() <= 0:
            return {}

        candidate_scores_raw = scores.index_select(0, candidate_idx_t).detach().float()
        if score_order == "lowest":
            # Keep "larger is better" semantics for observation even if retrieval is reversed.
            candidate_scores = -candidate_scores_raw
        else:
            candidate_scores = candidate_scores_raw

        candidate_n = int(candidate_scores.numel())
        if candidate_n <= 0:
            return {}

        probe_n = min(max(1, int(recent_probe_n)), candidate_n)
        top_m = min(max(1, int(self.config.get("observation_top_m", 2))), candidate_n)
        recent_scores = candidate_scores[-probe_n:]
        history_scores = candidate_scores[:-probe_n]

        recent_top_mean = self._mean_top_values(recent_scores, top_m)
        history_top_mean = self._mean_top_values(history_scores, top_m)
        candidate_std = float(candidate_scores.std(unbiased=False).detach().cpu().item())
        eps = 1e-6
        if math.isnan(history_top_mean):
            recent_advantage_z = 0.0
        else:
            recent_advantage_z = (recent_top_mean - history_top_mean) / max(candidate_std, eps)

        if candidate_n <= 1:
            uncertainty = 0.0
            top_margin_z = 0.0
        else:
            if candidate_std <= eps:
                uncertainty = 1.0
                top_margin_z = 0.0
            else:
                z_scores = (candidate_scores - candidate_scores.mean()) / max(candidate_std, eps)
                probs = torch.softmax(z_scores, dim=0)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
                uncertainty = float((entropy / math.log(candidate_n)).detach().cpu().item())
                top2 = torch.topk(candidate_scores, k=2, largest=True).values
                top_margin = (top2[0] - top2[1]) / max(candidate_std, eps)
                top_margin_z = float(top_margin.detach().cpu().item())

        return {
            "recent_score_advantage_z": float(recent_advantage_z),
            "retrieval_uncertainty": float(uncertainty),
            "recent_top_mean": float(recent_top_mean),
            "history_top_mean": (
                float(history_top_mean) if not math.isnan(history_top_mean) else None
            ),
            "candidate_score_std": float(candidate_std),
            "top_margin_z": float(top_margin_z),
            "recent_probe_frames": int(probe_n),
            "candidate_frames": int(candidate_n),
            "top_m": int(top_m),
        }

    def _safe_observation_name(self, value: Any, default: str) -> str:
        text = str(value or "").strip()
        if not text:
            text = default
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        return text or default

    def _observation_task_name(self) -> str:
        raw = str(os.environ.get("SHALLOWSTREAM_V3_OBS_TASK", "")).strip()
        if not raw:
            raw = str(os.environ.get("TASK", "")).strip()
        if not raw:
            raw = "unknown"
        if raw.lower() == "contextual":
            raw = "contexual"
        return self._safe_observation_name(raw.lower(), "unknown")

    def _observation_run_tag(self) -> str:
        return self._safe_observation_name(
            os.environ.get("SHALLOWSTREAM_V3_OBS_RUN_TAG", "manual"),
            "manual",
        )

    def _observation_output_dir(self) -> str:
        root = str(os.environ.get("SHALLOWSTREAM_V3_OBS_DIR", "")).strip()
        if not root:
            root = str(
                self.config.get(
                    "observation_dir",
                    "./outputs/streamingbench/observations/shallowstream_v3",
                )
            )
        return os.path.abspath(os.path.join(root, self._observation_run_tag()))

    def _write_json_atomic(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)

    def _read_json_dict(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(
                f"[{self.log_name}-Obs] ignored invalid summary file {path}: {exc}",
                flush=True,
            )
        return {}

    def _record_retrieval_observation_metrics(self, metrics: Dict[str, Any]) -> None:
        if not bool(self.config.get("observation_enabled", False)):
            return
        if not bool(self.config.get("observation_save", True)):
            return
        if not isinstance(metrics, dict) or not metrics:
            return

        numeric_items: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                value_f = float(value)
                if math.isfinite(value_f):
                    numeric_items[str(key)] = value_f
        if not numeric_items:
            return

        acc = self._observation_accumulator
        acc["count"] = int(acc.get("count", 0)) + 1
        sums = acc.setdefault("sums", {})
        mins = acc.setdefault("mins", {})
        maxs = acc.setdefault("maxs", {})
        for key, value_f in numeric_items.items():
            sums[key] = float(sums.get(key, 0.0)) + value_f
            mins[key] = value_f if key not in mins else min(float(mins[key]), value_f)
            maxs[key] = value_f if key not in maxs else max(float(maxs[key]), value_f)

    def _build_observation_summary(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        tasks_raw = raw_payload.get("tasks", {})
        if not isinstance(tasks_raw, dict):
            tasks_raw = {}

        tasks_summary: Dict[str, Any] = {}
        for task, entry in sorted(tasks_raw.items()):
            if not isinstance(entry, dict):
                continue
            count = int(entry.get("count", 0))
            sums = entry.get("sums", {})
            if count <= 0 or not isinstance(sums, dict):
                continue
            averages = {
                str(key): float(value) / float(count)
                for key, value in sorted(sums.items())
                if isinstance(value, (int, float))
            }
            tasks_summary[str(task)] = {
                "count": count,
                "averages": averages,
                "mins": entry.get("mins", {}),
                "maxs": entry.get("maxs", {}),
                "last_updated": entry.get("last_updated"),
            }

        return {
            "run_tag": raw_payload.get("run_tag", self._observation_run_tag()),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tasks": tasks_summary,
        }

    def _write_observation_summary_csv(self, path: str, summary: Dict[str, Any]) -> None:
        tasks = summary.get("tasks", {})
        if not isinstance(tasks, dict):
            return
        metric_keys = sorted(
            {
                str(key)
                for entry in tasks.values()
                if isinstance(entry, dict)
                for key in (entry.get("averages", {}) or {}).keys()
            }
        )
        tmp_path = f"{path}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "count"] + metric_keys)
            writer.writeheader()
            for task, entry in sorted(tasks.items()):
                averages = entry.get("averages", {}) if isinstance(entry, dict) else {}
                row = {
                    "task": task,
                    "count": int(entry.get("count", 0)) if isinstance(entry, dict) else 0,
                }
                for key in metric_keys:
                    value = averages.get(key)
                    row[key] = "" if value is None else f"{float(value):.6f}"
                writer.writerow(row)
        os.replace(tmp_path, path)

    def _flush_retrieval_observation_summary(self) -> None:
        if not bool(self.config.get("observation_enabled", False)):
            return
        if self._observation_summary_flushed:
            return
        self._observation_summary_flushed = True
        if not bool(self.config.get("observation_save", True)):
            return

        local_count = int(self._observation_accumulator.get("count", 0))
        if local_count <= 0:
            return

        out_dir = self._observation_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        raw_path = os.path.join(out_dir, "summary_by_task_raw.json")
        summary_path = os.path.join(out_dir, "summary_by_task.json")
        csv_path = os.path.join(out_dir, "summary_by_task.csv")
        task = self._observation_task_name()

        raw_payload = self._read_json_dict(raw_path)
        if not raw_payload:
            raw_payload = {
                "run_tag": self._observation_run_tag(),
                "output_dir": out_dir,
                "tasks": {},
            }
        tasks = raw_payload.setdefault("tasks", {})
        if not isinstance(tasks, dict):
            tasks = {}
            raw_payload["tasks"] = tasks

        entry = tasks.setdefault(task, {"count": 0, "sums": {}, "mins": {}, "maxs": {}})
        entry["count"] = int(entry.get("count", 0)) + local_count

        for bucket_name in ("sums", "mins", "maxs"):
            entry.setdefault(bucket_name, {})
        for key, value in (self._observation_accumulator.get("sums", {}) or {}).items():
            entry["sums"][key] = float(entry["sums"].get(key, 0.0)) + float(value)
        for key, value in (self._observation_accumulator.get("mins", {}) or {}).items():
            entry["mins"][key] = (
                float(value)
                if key not in entry["mins"]
                else min(float(entry["mins"][key]), float(value))
            )
        for key, value in (self._observation_accumulator.get("maxs", {}) or {}).items():
            entry["maxs"][key] = (
                float(value)
                if key not in entry["maxs"]
                else max(float(entry["maxs"][key]), float(value))
            )

        entry["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        raw_payload["updated_at"] = entry["last_updated"]

        summary = self._build_observation_summary(raw_payload)
        task_summary = {
            "run_tag": summary["run_tag"],
            "updated_at": summary["updated_at"],
            "task": task,
            **summary["tasks"].get(task, {}),
        }

        self._write_json_atomic(raw_path, raw_payload)
        self._write_json_atomic(summary_path, summary)
        self._write_json_atomic(
            os.path.join(out_dir, f"{task}_observation_summary.json"),
            task_summary,
        )
        self._write_observation_summary_csv(csv_path, summary)

    def _observe_retrieval(
        self,
        *,
        scores: torch.Tensor,
        candidate_idx_t: torch.Tensor,
        recent_n: int,
        score_order: str,
        video_path: Optional[str],
    ) -> None:
        super()._observe_retrieval(
            scores=scores,
            candidate_idx_t=candidate_idx_t,
            recent_n=recent_n,
            score_order=score_order,
            video_path=video_path,
        )
        if not bool(self.config.get("observation_enabled", False)):
            return
        observation_recent_n = int(
            self.config.get("observation_recent_probe_frames", recent_n or 4)
        )
        observation_metrics = self._compute_retrieval_observation_metrics(
            scores=scores,
            candidate_idx_t=candidate_idx_t,
            recent_probe_n=observation_recent_n,
            score_order=score_order,
        )
        self._record_retrieval_observation_metrics(observation_metrics)
        if not bool(self.config.get("observation_print", True)) or not observation_metrics:
            return

        obs_payload = {
            "recent_score_advantage_z": round(
                float(observation_metrics["recent_score_advantage_z"]), 4
            ),
            "retrieval_uncertainty": round(
                float(observation_metrics["retrieval_uncertainty"]), 4
            ),
            "recent_top_mean": round(
                float(observation_metrics["recent_top_mean"]), 4
            ),
            "history_top_mean": (
                round(float(observation_metrics["history_top_mean"]), 4)
                if observation_metrics.get("history_top_mean") is not None
                else None
            ),
            "candidate_score_std": round(
                float(observation_metrics["candidate_score_std"]), 4
            ),
            "top_margin_z": round(float(observation_metrics["top_margin_z"]), 4),
            "recent_probe_frames": int(observation_metrics["recent_probe_frames"]),
            "candidate_frames": int(observation_metrics["candidate_frames"]),
        }
        if video_path:
            obs_payload["chunk"] = os.path.basename(str(video_path))
        print(
            f"[{self.log_name}-Obs] "
            + json.dumps(obs_payload, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
