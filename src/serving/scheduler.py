"""Simple FIFO and fixed-batch request scheduling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class BatchResult:
    batch_size: int
    queue_wait_ms: float
    batch_latency_ms: float
    request_ids: tuple[str, ...]


class FIFOScheduler:
    def __init__(self, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.queue: deque[Request] = deque()

    def submit(self, request: Request) -> None:
        self.queue.append(request)

    def submit_many(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.submit(request)

    def next_batch(self) -> list[Request]:
        batch = []
        while self.queue and len(batch) < self.batch_size:
            batch.append(self.queue.popleft())
        return batch

    def drain(self) -> list[list[Request]]:
        batches = []
        while self.queue:
            batches.append(self.next_batch())
        return batches


def append_scheduler_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-size request batching")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/03_batching/runs.csv"),
    )
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
    ).to(device)
    model.eval()

    for _ in range(args.warmup):
        warmup_ids = tokenizer(
            ["Large language model inference"],
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            model.generate(
                input_ids=warmup_ids["input_ids"].to(device),
                attention_mask=warmup_ids["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()

    scheduler = FIFOScheduler(args.batch_size)
    submitted_at = time.perf_counter()
    requests = [
        Request(
            request_id=f"{uuid.uuid4().hex[:8]}-{index}",
            prompt="Large language model inference",
            max_new_tokens=args.max_new_tokens,
        )
        for index in range(args.requests)
    ]
    scheduler.submit_many(requests)
    batches = scheduler.drain()

    rows: list[dict[str, object]] = []
    batch_results: list[BatchResult] = []
    completed_tokens = 0
    benchmark_start = time.perf_counter()

    for batch in batches:
        if not batch:
            continue
        now = time.perf_counter()
        queue_wait_ms = (now - submitted_at) * 1000
        prompts = [request.prompt for request in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        batch_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        batch_latency_ms = (time.perf_counter() - batch_start) * 1000
        for request, sequence in zip(batch, output_ids):
            completed_tokens += len(sequence) - input_ids.shape[1]

        result = BatchResult(
            batch_size=len(batch),
            queue_wait_ms=queue_wait_ms,
            batch_latency_ms=batch_latency_ms,
            request_ids=tuple(request.request_id for request in batch),
        )
        batch_results.append(result)

        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": "03_batching",
                "model": args.model,
                "device": device,
                "batch_size": args.batch_size,
                "requests": args.requests,
                "actual_batch_size": len(batch),
                "queue_wait_ms": queue_wait_ms,
                "batch_latency_ms": batch_latency_ms,
                "completed_tokens": completed_tokens,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if device == "cuda"
                    else None
                ),
            }
        )

    benchmark_seconds = time.perf_counter() - benchmark_start
    request_throughput = args.requests / benchmark_seconds
    token_throughput = completed_tokens / benchmark_seconds
    batch_latencies = [result.batch_latency_ms for result in batch_results]

    def percentile(values: list[float], percentage: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * percentage / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    summary = {
        "batch_size": args.batch_size,
        "requests": args.requests,
        "completed_requests": args.requests,
        "completed_tokens": completed_tokens,
        "request_throughput_rps": request_throughput,
        "token_throughput_tps": token_throughput,
        "batch_latency_ms_p50": percentile(batch_latencies, 50),
        "batch_latency_ms_p95": percentile(batch_latencies, 95),
        "batch_latency_ms_p99": percentile(batch_latencies, 99),
    }
    append_scheduler_csv(args.csv, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()