"""Bounded, acknowledgement-driven cache coordination across processes."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Sequence, TypeVar


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT", bound=Mapping[str, Any])


def per_worker_thread_count(
    total_threads: int,
    workers: int,
    *,
    maximum: int = 4,
) -> int:
    """Divide one process CPU budget without letting libraries oversubscribe."""
    if total_threads <= 0 or workers <= 0 or maximum <= 0:
        raise ValueError("thread budget, worker count, and maximum must be positive")
    return max(1, min(maximum, total_threads // workers))


def _validated_token(value: str, label: str) -> str:
    token = str(value).strip()
    if not _SAFE_TOKEN.fullmatch(token):
        raise ValueError(
            f"Invalid {label} {value!r}; use letters, digits, dot, dash, or underscore"
        )
    return token


def cache_entry_ack_path(state_dir: str, cache_key: str, consumer_id: str) -> Path:
    key = _validated_token(cache_key, "routed cache key")
    consumer = _validated_token(consumer_id, "routed cache consumer id")
    return Path(state_dir) / "acks" / key / f"{consumer}.ack"


def cache_consumer_cancel_path(state_dir: str, consumer_id: str) -> Path:
    consumer = _validated_token(consumer_id, "routed cache consumer id")
    return Path(state_dir) / "cancelled_consumers" / f"{consumer}.cancelled"


def cache_consumer_progress_path(state_dir: str, consumer_id: str) -> Path:
    consumer = _validated_token(consumer_id, "routed cache consumer id")
    return Path(state_dir) / "consumer_progress" / f"{consumer}.progress"


def mark_cache_consumer_progress(state_dir: str, consumer_id: str) -> Path:
    """Publish a lightweight progress lease for one live consumer."""
    marker = cache_consumer_progress_path(state_dir, consumer_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o644)
    os.close(descriptor)
    os.utime(marker, None)
    return marker


def cache_consumer_progress_is_fresh(
    state_dir: str,
    consumer_id: str,
    timeout_seconds: float,
    *,
    now: Optional[float] = None,
) -> bool:
    marker = cache_consumer_progress_path(state_dir, consumer_id)
    try:
        modified_at = marker.stat().st_mtime
    except FileNotFoundError:
        return False
    observed_at = time.time() if now is None else float(now)
    return observed_at - modified_at <= float(timeout_seconds)


def acknowledge_cache_entry(state_dir: str, cache_key: str, consumer_id: str) -> Path:
    marker = cache_entry_ack_path(state_dir, cache_key, consumer_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    else:
        os.close(descriptor)
    mark_cache_consumer_progress(state_dir, consumer_id)
    return marker


def cancel_cache_consumer(state_dir: str, consumer_id: str) -> Path:
    """Release future leases after the orchestrator observes consumer exit."""
    marker = cache_consumer_cancel_path(state_dir, consumer_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return marker
    os.close(descriptor)
    return marker


def missing_cache_entry_consumers(
    state_dir: str,
    cache_key: str,
    consumer_ids: Sequence[str],
) -> list[str]:
    if not consumer_ids:
        raise ValueError("routed cache entries require at least one consumer")
    return [
        consumer_id
        for consumer_id in consumer_ids
        if not cache_entry_ack_path(state_dir, cache_key, consumer_id).is_file()
        and not cache_consumer_cancel_path(state_dir, consumer_id).is_file()
    ]


def cache_entry_acknowledged_by_all(
    state_dir: str,
    cache_key: str,
    consumer_ids: Sequence[str],
) -> bool:
    return not missing_cache_entry_consumers(state_dir, cache_key, consumer_ids)


def clear_cache_entry_acknowledgements(state_dir: str, cache_key: str) -> None:
    key = _validated_token(cache_key, "routed cache key")
    directory = Path(state_dir) / "acks" / key
    if not directory.is_dir():
        return
    for marker in directory.iterdir():
        if marker.is_file():
            marker.unlink()
    directory.rmdir()


@dataclass(frozen=True)
class RoutedCacheStats:
    published_entries: int
    evicted_entries: int
    peak_outstanding_entries: int


@dataclass
class _Published(Generic[ItemT, ResultT]):
    item: ItemT
    result: ResultT
    cache_key: str
    consumers: tuple[str, ...]
    published_at: float


def run_bounded_routed_cache(
    *,
    items: Sequence[ItemT],
    produce: Callable[[ItemT], ResultT],
    consumers_for: Callable[[ItemT], Sequence[str]],
    cache_key_for: Callable[[ItemT, ResultT], str],
    evict: Callable[[ItemT, ResultT], None],
    state_dir: str,
    workers: int,
    max_outstanding_entries: int,
    poll_seconds: float,
    ack_timeout_seconds: float,
    stop_event: Any = None,
    on_published: Optional[Callable[[ItemT, ResultT, RoutedCacheStats], None]] = None,
    on_evicted: Optional[Callable[[ItemT, ResultT, RoutedCacheStats], None]] = None,
) -> RoutedCacheStats:
    """Produce into a bounded pool and release any fully acknowledged entry.

    Capacity counts active productions plus published, unacknowledged entries.
    Entries are independent: a slow consumer cannot pin unrelated later entries.
    The acknowledgement timeout measures consumer inactivity, not artifact age:
    a lane that continues completing other work units keeps its pending leases
    alive, while a genuinely stalled lane remains bounded.
    """
    if workers <= 0:
        raise ValueError("routed cache workers must be positive")
    if max_outstanding_entries <= 0:
        raise ValueError("routed cache capacity must be positive")
    if poll_seconds <= 0 or ack_timeout_seconds <= 0:
        raise ValueError("routed cache poll and acknowledgement timeouts must be positive")
    if not state_dir:
        raise ValueError("routed cache state directory is required")

    ordered_items = list(items)
    consumers_by_index: Dict[int, tuple[str, ...]] = {}
    for index, item in enumerate(ordered_items):
        consumers = tuple(
            dict.fromkeys(
                _validated_token(str(value), "routed cache consumer id")
                for value in consumers_for(item)
            )
        )
        if not consumers:
            raise ValueError(f"routed cache item {index} has no consumers")
        consumers_by_index[index] = consumers

    published_count = evicted_count = peak_outstanding = 0

    def stats() -> RoutedCacheStats:
        return RoutedCacheStats(
            published_entries=published_count,
            evicted_entries=evicted_count,
            peak_outstanding_entries=peak_outstanding,
        )

    active: Dict[Future[ResultT], tuple[int, ItemT]] = {}
    published: Dict[int, _Published[ItemT, ResultT]] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="routed-cache") as executor:
        while next_index < len(ordered_items) or active or published:
            if stop_event is not None and stop_event.is_set():
                for future in active:
                    future.cancel()
                raise RuntimeError("routed cache producer was cancelled")

            for index, entry in list(published.items()):
                if not cache_entry_acknowledged_by_all(
                    state_dir,
                    entry.cache_key,
                    entry.consumers,
                ):
                    continue
                evict(entry.item, entry.result)
                # Keep acknowledgement receipts for the lifetime of this
                # unique route run. Consumers can legitimately send an
                # idempotent late acknowledgement after the producer observes
                # the first one; deleting the directory here races that write
                # and can fail with ENOTEMPTY on shared filesystems.
                del published[index]
                evicted_count += 1
                if on_evicted is not None:
                    on_evicted(entry.item, entry.result, stats())

            while (
                next_index < len(ordered_items)
                and len(active) < workers
                and len(active) + len(published) < max_outstanding_entries
            ):
                item = ordered_items[next_index]
                future = executor.submit(produce, item)
                active[future] = (next_index, item)
                next_index += 1
                peak_outstanding = max(
                    peak_outstanding,
                    len(active) + len(published),
                )

            if active:
                done, _pending = wait(
                    active,
                    timeout=poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    index, item = active.pop(future)
                    result = future.result()
                    cache_key = _validated_token(
                        cache_key_for(item, result),
                        "routed cache key",
                    )
                    published[index] = _Published(
                        item=item,
                        result=result,
                        cache_key=cache_key,
                        consumers=consumers_by_index[index],
                        published_at=time.monotonic(),
                    )
                    published_count += 1
                    if on_published is not None:
                        on_published(item, result, stats())
            elif published:
                time.sleep(poll_seconds)

            now = time.monotonic()
            expired = [
                (index, entry)
                for index, entry in published.items()
                if now - entry.published_at > ack_timeout_seconds
            ]
            if expired:
                index, entry = min(expired, key=lambda value: value[1].published_at)
                missing = missing_cache_entry_consumers(
                    state_dir,
                    entry.cache_key,
                    entry.consumers,
                )
                wall_now = time.time()
                stale = [
                    consumer
                    for consumer in missing
                    if not cache_consumer_progress_is_fresh(
                        state_dir,
                        consumer,
                        ack_timeout_seconds,
                        now=wall_now,
                    )
                ]
                if not stale:
                    entry.published_at = now
                    continue
                raise TimeoutError(
                    "Timed out waiting for routed cache acknowledgements: "
                    f"item_index={index}, cache_key={entry.cache_key}, "
                    f"missing_consumers={missing}, "
                    f"stale_consumers={stale}, "
                    f"timeout_seconds={ack_timeout_seconds}"
                )

    return stats()
