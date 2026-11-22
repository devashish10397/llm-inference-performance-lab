"""Helpers for summarizing and recording benchmark measurements."""

from __future__ import annotations

import math
import csv
import platform
import uuid
import torch
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Any

if TYPE_CHECKING:
    from src.inference.minimal_decoder import GenerationMetrics


def percentile(values: Iterable[float], percentile: float) -> float:
    """Return a linear-interpolated percentile for a non-empty value list."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("values must not be empty")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "experiment",
    "repeat",
    "model",
    "device",
    "precision",
    "python_version",
    "torch_version",
    "gpu_name",
    "max_new_tokens",
    "kv_cache_enabled",
    "estimated_kv_cache_bytes",
    "prompt",
    "prompt_tokens",
    "generated_tokens",
    "prefill_ms",
    "ttft_ms",
    "tpot_ms",
    "end_to_end_ms",
    "throughput_tokens_per_second",
    "peak_gpu_memory_mb",
]


def append_benchmark_csv(
    path: str | Path,
    metrics: "GenerationMetrics",
    experiment: str = "01_baseline",
    repeat: int = 1,
) -> Path:
    """Append one benchmark run to a stable, analysis-friendly CSV."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    existing_rows: list[dict[str, Any]] = []

    if not is_new:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != CSV_COLUMNS:
                for row in reader:
                    if None not in row:
                        existing_rows.append(
                            {column: row.get(column) for column in reader.fieldnames or []}
                        )
                with csv_path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow(
                            {column: row.get(column) for column in CSV_COLUMNS}
                        )

    throughput = (
        metrics.generated_tokens / (metrics.end_to_end_ms / 1000)
        if metrics.end_to_end_ms > 0
        else 0.0
    )
    precision = "float16" if metrics.device == "cuda" else "float32"
    row: Mapping[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "repeat": repeat,
        "model": metrics.model,
        "device": metrics.device,
        "precision": precision,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "max_new_tokens": getattr(metrics, "max_new_tokens", None),
        "kv_cache_enabled": getattr(metrics, "kv_cache_enabled", True),
        "estimated_kv_cache_bytes": getattr(
            metrics,
            "estimated_kv_cache_bytes",
            None,
        ),
        "prompt": metrics.prompt,
        "prompt_tokens": metrics.prompt_tokens,
        "generated_tokens": metrics.generated_tokens,
        "prefill_ms": metrics.prefill_ms,
        "ttft_ms": metrics.ttft_ms,
        "tpot_ms": metrics.tpot_ms,
        "end_to_end_ms": metrics.end_to_end_ms,
        "throughput_tokens_per_second": throughput,
        "peak_gpu_memory_mb": metrics.peak_gpu_memory_mb,
    }

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return csv_path