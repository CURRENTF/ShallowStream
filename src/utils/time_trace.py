import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


_SAMPLE_ID_RE = re.compile(r"sample_(\d+)", re.IGNORECASE)
_CLIP_RANGE_RE = re.compile(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.[A-Za-z0-9]+$")


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("._-")
    return value or "trace"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="milliseconds")


def extract_sample_id(*paths: Optional[str]) -> Optional[int]:
    for raw in paths:
        if not raw:
            continue
        matched = _SAMPLE_ID_RE.search(str(raw))
        if matched:
            try:
                return int(matched.group(1))
            except ValueError:
                continue
    return None


def extract_clip_range_seconds(path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not path:
        return None, None
    base = os.path.basename(str(path))
    matched = _CLIP_RANGE_RE.search(base)
    if not matched:
        return None, None
    try:
        return float(matched.group(1)), float(matched.group(2))
    except ValueError:
        return None, None


def make_prompt_preview(text: str, max_len: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


class StageRecorder:
    def __init__(self) -> None:
        self._stages: List[Dict[str, Any]] = []

    def start(self, name: str) -> Dict[str, Any]:
        return {
            "name": str(name),
            "start_perf": time.perf_counter(),
            "start_ts": time.time(),
        }

    def end(self, token: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        end_perf = time.perf_counter()
        end_ts = time.time()
        start_perf = float(token.get("start_perf", end_perf))
        start_ts = float(token.get("start_ts", end_ts))
        stage = {
            "name": str(token.get("name", "unknown")),
            "start_ts": start_ts,
            "start_iso": _iso(start_ts),
            "end_ts": end_ts,
            "end_iso": _iso(end_ts),
            "duration_ms": max(0.0, (end_perf - start_perf) * 1000.0),
        }
        if extra:
            stage.update(extra)
        self._stages.append(stage)
        return stage

    def durations_ms(self) -> Dict[str, float]:
        return {stage["name"]: float(stage["duration_ms"]) for stage in self._stages}

    def stages(self) -> List[Dict[str, Any]]:
        return list(self._stages)


class TimeTraceWriter:
    _lock = threading.Lock()

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        model_name: str,
        run_tag: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = os.path.abspath(str(output_dir))
        self.model_name = str(model_name)
        self.run_tag = str(run_tag or "")
        self.path: Optional[str] = None

        if not self.enabled:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = _slugify(self.model_name)
        if self.run_tag:
            stem = f"{stem}_{_slugify(self.run_tag)}"
        self.path = os.path.join(self.output_dir, f"{stem}_{ts}.jsonl")

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or not self.path:
            return
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
