"""Plot benchmark latency and memory measurements from a runs CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark(csv_path: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(csv_path)
    required = {"experiment", "repeat", "ttft_ms", "tpot_ms", "peak_gpu_memory_mb"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 5))
    grouped = data.groupby("experiment")[["ttft_ms", "tpot_ms"]].median()
    grouped.plot(kind="bar", ax=axis)
    axis.set_title("Median TTFT and TPOT")
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Milliseconds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    latency_path = output_dir / "latency_comparison.png"
    fig.savefig(latency_path, dpi=160)
    plt.close(fig)
    generated.append(latency_path)

    if data["peak_gpu_memory_mb"].notna().any():
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(
            data["prompt_tokens"],
            data["peak_gpu_memory_mb"],
            s=45,
            alpha=0.75,
        )
        axis.set_title("Prompt Length vs. Peak GPU Memory")
        axis.set_xlabel("Prompt tokens")
        axis.set_ylabel("Peak GPU memory (MB)")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        memory_path = output_dir / "gpu_memory.png"
        fig.savefig(memory_path, dpi=160)
        plt.close(fig)
        generated.append(memory_path)

    return generated


def plot_kv_cache(csv_path: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(csv_path)
    required = {"kv_cache_enabled", "tpot_ms", "ttft_ms", "end_to_end_ms", "peak_gpu_memory_mb"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"KV-cache CSV is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data["cache"] = data["kv_cache_enabled"].map({True: "on", False: "off"})
    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 5))
    summary = data.groupby("cache")[["ttft_ms", "tpot_ms", "end_to_end_ms"]].median()
    summary.plot(kind="bar", ax=axis)
    axis.set_title("KV Cache: Median Latency")
    axis.set_xlabel("KV cache")
    axis.set_ylabel("Milliseconds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    latency_path = output_dir / "kv_cache_latency.png"
    fig.savefig(latency_path, dpi=160)
    plt.close(fig)
    generated.append(latency_path)

    fig, axis = plt.subplots(figsize=(8, 5))
    memory = data.groupby("cache")["peak_gpu_memory_mb"].median()
    memory.plot(kind="bar", ax=axis)
    axis.set_title("KV Cache: Median Peak GPU Memory")
    axis.set_xlabel("KV cache")
    axis.set_ylabel("Peak GPU memory (MB)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    memory_path = output_dir / "kv_cache_memory.png"
    fig.savefig(memory_path, dpi=160)
    plt.close(fig)
    generated.append(memory_path)

    return generated


def plot_batching(csv_path: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(csv_path)
    required = {"batch_size", "batch_latency_ms", "queue_wait_ms"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Batching CSV is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 5))
    grouped = data.groupby("batch_size")["batch_latency_ms"].median()
    grouped.plot(kind="bar", ax=axis)
    axis.set_title("Fixed Batching: Median Batch Latency")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Milliseconds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    latency_path = output_dir / "batch_latency.png"
    fig.savefig(latency_path, dpi=160)
    plt.close(fig)
    generated.append(latency_path)

    fig, axis = plt.subplots(figsize=(8, 5))
    queue = data.groupby("batch_size")["queue_wait_ms"].median()
    queue.plot(kind="bar", ax=axis)
    axis.set_title("Fixed Batching: Median Queue Wait")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Milliseconds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    queue_path = output_dir / "batch_queue_wait.png"
    fig.savefig(queue_path, dpi=160)
    plt.close(fig)
    generated.append(queue_path)

    return generated


def plot_vllm(json_path: Path, output_dir: Path) -> list[Path]:
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for benchmark in data["benchmarks"]:
        metrics = benchmark["metrics"]
        streams = benchmark["config"]["strategy"]["streams"]
        rows.append(
            {
                "concurrency": streams,
                "ttft_ms": metrics["time_to_first_token_ms"]["successful"]["median"],
                "tpot_ms": metrics["time_per_output_token_ms"]["successful"]["median"],
                "output_tokens_per_second": metrics["output_tokens_per_second"]["successful"]["mean"],
                "requests_per_second": metrics["requests_per_second"]["successful"]["mean"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("concurrency")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        frame["concurrency"],
        frame["ttft_ms"],
        marker="o",
        linewidth=2,
    )
    axis.set_title("vLLM: Median TTFT vs. Concurrency")
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("TTFT (ms)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    ttft_path = output_dir / "vllm_ttft.png"
    fig.savefig(ttft_path, dpi=160)
    plt.close(fig)
    generated.append(ttft_path)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        frame["concurrency"],
        frame["output_tokens_per_second"],
        marker="o",
        linewidth=2,
        label="Output tokens/s",
    )
    axis.plot(
        frame["concurrency"],
        frame["requests_per_second"],
        marker="o",
        linewidth=2,
        label="Requests/s",
    )
    axis.set_title("vLLM: Throughput vs. Concurrency")
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Throughput")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    throughput_path = output_dir / "vllm_throughput.png"
    fig.savefig(throughput_path, dpi=160)
    plt.close(fig)
    generated.append(throughput_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Create charts from benchmark runs")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("results/01_baseline/runs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/01_baseline/charts"),
    )
    parser.add_argument(
        "--kv-cache",
        type=Path,
        default=Path("results/02_kv_cache/runs.csv"),
    )
    parser.add_argument(
        "--batching",
        type=Path,
        default=Path("results/03_batching/runs.csv"),
    )
    parser.add_argument(
        "--vllm",
        type=Path,
        default=Path("results/04_vllm/benchmarks.json"),
    )
    parser.add_argument("--all-experiments", action="store_true")
    args = parser.parse_args()
    for chart in plot_benchmark(args.csv, args.output):
        print(chart)
    if args.all_experiments:
        if args.kv_cache.exists():
            for chart in plot_kv_cache(
                args.kv_cache,
                args.kv_cache.parent / "charts",
            ):
                print(chart)
        if args.batching.exists():
            for chart in plot_batching(
                args.batching,
                args.batching.parent / "charts",
            ):
                print(chart)
        if args.vllm.exists():
            for chart in plot_vllm(
                args.vllm,
                args.vllm.parent / "charts",
            ):
                print(chart)


if __name__ == "__main__":
    main()