"""Plot benchmark latency and memory measurements from a runs CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args()
    for chart in plot_benchmark(args.csv, args.output):
        print(chart)


if __name__ == "__main__":
    main()