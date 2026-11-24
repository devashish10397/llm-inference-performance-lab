"""Generate normalized summaries and charts for one benchmark machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def percentile(values: pd.Series, percent: float) -> float:
    return float(values.quantile(percent / 100))


def hardware_label(hardware: dict) -> str:
    software = hardware.get("software", {})
    return (
        f"{hardware['gpu']} / {hardware['gpu_vram_gib']:.2f} GiB; "
        f"{hardware['cpu']}; {hardware['ram_gib']:.2f} GiB RAM; "
        f"{hardware['os']} {hardware['os_version']}; "
        f"NVIDIA driver {hardware['nvidia_driver']}; "
        f"compute capability {hardware['compute_capability']}; "
        f"CUDA/PyTorch {software.get('cuda','')}/{software.get('torch','')}; "
        f"vLLM/GuideLLM {software.get('vllm','')}/{software.get('guidellm','')}; "
        f"model {software.get('model','')}; {software.get('precision','')}; "
        f"workloads direct {software.get('baseline_prompt_output','')}, "
        f"KV {software.get('kv_prompt_output','')}, "
        f"vLLM {software.get('vllm_prompt_output','')}; "
        f"warm-ups/repeats {software.get('baseline_warmup_repeats','')}, "
        f"{software.get('kv_warmup_repeats','')}"
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_baseline(root: Path, hardware: dict) -> pd.DataFrame:
    data = pd.read_csv(root / "baseline/runs.csv")
    summary = pd.DataFrame(
        [{
            "machine": root.name,
            "model": data["model"].iloc[0],
            "device": data["device"].iloc[0],
            "prompt_tokens": int(data["prompt_tokens"].median()),
            "output_tokens": int(data["generated_tokens"].median()),
            "ttft_p50_ms": percentile(data["ttft_ms"], 50),
            "ttft_p95_ms": percentile(data["ttft_ms"], 95),
            "ttft_p99_ms": percentile(data["ttft_ms"], 99),
            "tpot_p50_ms": percentile(data["tpot_ms"], 50),
            "tpot_p95_ms": percentile(data["tpot_ms"], 95),
            "tpot_p99_ms": percentile(data["tpot_ms"], 99),
            "end_to_end_p50_ms": percentile(data["end_to_end_ms"], 50),
            "output_tps_p50": percentile(data["throughput_tokens_per_second"], 50),
            "peak_gpu_memory_mib": percentile(data["peak_gpu_memory_mb"], 50),
        }]
    )
    summary.to_csv(root / "baseline/summary.csv", index=False)

    label = hardware_label(hardware)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        ["TTFT P50", "TPOT P50", "End-to-end P50"],
        [
            summary["ttft_p50_ms"].iloc[0],
            summary["tpot_p50_ms"].iloc[0],
            summary["end_to_end_p50_ms"].iloc[0],
        ],
    )
    axis.set_ylabel("Milliseconds")
    axis.set_title("GM700 GPT-2 Baseline Latency")
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "baseline/latency_comparison.png")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(["Peak GPU memory"], [summary["peak_gpu_memory_mib"].iloc[0]])
    axis.set_ylabel("MiB")
    axis.set_title("GM700 GPT-2 Peak GPU Memory")
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "baseline/gpu_memory.png")
    return summary


def generate_kv_cache(root: Path, hardware: dict) -> pd.DataFrame:
    data = pd.read_csv(root / "kv-cache/runs.csv")
    rows = []
    for (output_tokens, cache_enabled), group in data.groupby(
        ["max_new_tokens", "kv_cache_enabled"]
    ):
        mean_e2e = float(group["end_to_end_ms"].mean())
        rows.append({
            "machine": root.name,
            "model": group["model"].iloc[0],
            "prompt_tokens": int(group["prompt_tokens"].median()),
            "output_tokens": int(output_tokens),
            "cache": "on" if bool(cache_enabled) else "off",
            "ttft_p50_ms": percentile(group["ttft_ms"], 50),
            "tpot_p50_ms": percentile(group["tpot_ms"], 50),
            "end_to_end_p50_ms": percentile(group["end_to_end_ms"], 50),
            "end_to_end_p95_ms": percentile(group["end_to_end_ms"], 95),
            "output_tps_p50": percentile(group["throughput_tokens_per_second"], 50),
            "peak_gpu_memory_mib": percentile(group["peak_gpu_memory_mb"], 50),
            "end_to_end_mean_ms": mean_e2e,
            "end_to_end_cv_pct": (
                float(group["end_to_end_ms"].std(ddof=1) / mean_e2e * 100)
                if mean_e2e
                else 0.0
            ),
        })
    summary = pd.DataFrame(rows).sort_values(["output_tokens", "cache"])
    summary.to_csv(root / "kv-cache/summary.csv", index=False)

    label = hardware_label(hardware)
    fig, axis = plt.subplots(figsize=(10, 6))
    pivot = summary.pivot(index="output_tokens", columns="cache", values="end_to_end_p50_ms")
    pivot.plot(kind="line", marker="o", ax=axis)
    axis.set_title("GM700 GPT-2 KV-Cache Scaling: Median End-to-End")
    axis.set_xlabel("Output tokens")
    axis.set_ylabel("Milliseconds")
    axis.grid(alpha=0.25)
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "kv-cache/kv_scaling.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    pivot = summary.pivot(index="output_tokens", columns="cache", values="end_to_end_cv_pct")
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("GM700 GPT-2 KV-Cache End-to-End Variance")
    axis.set_xlabel("Output tokens")
    axis.set_ylabel("Coefficient of variation (%)")
    axis.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "kv-cache/kv_variance.png")

    telemetry_path = root / "kv-cache/telemetry.csv"
    if telemetry_path.exists():
        telemetry = pd.read_csv(telemetry_path)
        for column in telemetry.columns:
            if column != "timestamp":
                telemetry[column] = pd.to_numeric(
                    telemetry[column].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(telemetry["temperature_c"], label="Temperature")
        axes[0].plot(telemetry["power_draw_w"], label="Power")
        axes[0].set_ylabel("C / W")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(telemetry["utilization_gpu_pct"], label="GPU utilization")
        axes[1].plot(telemetry["memory_used_mib"] / 1024, label="VRAM used")
        axes[1].set_ylabel("Percent / GiB")
        axes[1].set_xlabel("Telemetry sample")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.suptitle("GM700 GPT-2 KV-Cache GPU Telemetry")
        fig.text(0.5, 0.01, label, ha="center", fontsize=8)
        save_figure(fig, root / "kv-cache/gpu_telemetry.png")
    return summary


def generate_batching(root: Path, hardware: dict) -> pd.DataFrame:
    data = pd.read_csv(root / "batching/runs.csv")
    completed = data.groupby("batch_size")["completed_tokens"].max().rename("completed_tokens")
    elapsed = data.groupby("batch_size")["batch_latency_ms"].sum() / 1000
    summary = pd.DataFrame({
        "batch_size": completed.index,
        "completed_tokens": completed.values,
        "output_tps": (completed / elapsed.reindex(completed.index)).values,
        "batch_latency_p50_ms": [
            percentile(data.loc[data["batch_size"] == size, "batch_latency_ms"], 50)
            for size in completed.index
        ],
        "queue_wait_p50_ms": [
            percentile(data.loc[data["batch_size"] == size, "queue_wait_ms"], 50)
            for size in completed.index
        ],
    })
    summary.insert(0, "machine", root.name)
    summary.to_csv(root / "batching/summary.csv", index=False)

    label = hardware_label(hardware)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(summary["batch_size"], summary["output_tps"], marker="o")
    axis.set_title("GM700 GPT-2 Fixed-Batch Output Throughput")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Output tokens/s")
    axis.grid(alpha=0.25)
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "batching/batch_latency.png")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(summary["batch_size"], summary["queue_wait_p50_ms"], marker="o")
    axis.set_title("GM700 GPT-2 Fixed-Batch Median Queue Wait")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Milliseconds")
    axis.grid(alpha=0.25)
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "batching/batch_queue_wait.png")
    return summary


def generate_vllm(root: Path, hardware: dict) -> pd.DataFrame | None:
    json_path = root / "vllm/benchmarks.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for benchmark in data["benchmarks"]:
        metrics = benchmark["metrics"]
        rows.append({
            "machine": root.name,
            "concurrency": benchmark["config"]["strategy"]["streams"],
            "ttft_p50_ms": metrics["time_to_first_token_ms"]["successful"]["median"],
            "ttft_p95_ms": metrics["time_to_first_token_ms"]["successful"]["percentiles"]["p95"],
            "tpot_p50_ms": metrics["time_per_output_token_ms"]["successful"]["median"],
            "output_tps": metrics["output_tokens_per_second"]["successful"]["mean"],
            "requests_per_second": metrics["requests_per_second"]["successful"]["mean"],
            "successful": benchmark["config"].get("successful_requests", 0),
            "errored": 0,
        })
    summary = pd.DataFrame(rows).sort_values("concurrency")
    summary.to_csv(root / "vllm/summary.csv", index=False)

    label = hardware_label(hardware)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(summary["concurrency"], summary["ttft_p50_ms"], marker="o")
    axis.set_title("GM700 GPT-2 vLLM Median TTFT")
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Milliseconds")
    axis.grid(alpha=0.25)
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "vllm/vllm_ttft.png")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(summary["concurrency"], summary["output_tps"], marker="o", label="Output TPS")
    axis.plot(summary["concurrency"], summary["requests_per_second"], marker="o", label="Requests/s")
    axis.set_title("GM700 GPT-2 vLLM Throughput")
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Throughput")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.text(0.5, 0.01, label, ha="center", fontsize=8)
    save_figure(fig, root / "vllm/vllm_throughput.png")
    return summary


def generate_comparison(root: Path, hardware: dict) -> None:
    laptop_summary_path = root.parent / "rtx3050ti-laptop/comparison/summary.csv"
    if not laptop_summary_path.exists():
        return
    laptop = pd.read_csv(laptop_summary_path)

    gm_rows = []
    baseline = pd.read_csv(root / "baseline/summary.csv").iloc[0]
    gm_rows.append({
        "machine": "gm700-desktop",
        "scenario": "PyTorch baseline",
        "family": "4-token prompt",
        "prompt_tokens": 4,
        "output_tokens": 32,
        "concurrency_or_batch": 1,
        "ttft_ms": baseline["ttft_p50_ms"],
        "tpot_ms": baseline["tpot_p50_ms"],
        "output_tps": baseline["output_tps_p50"],
    })
    kv = pd.read_csv(root / "kv-cache/summary.csv")
    for cache in ("on", "off"):
        for output_tokens in (32, 64, 128):
            row = kv[
                (kv["cache"] == cache)
                & (kv["output_tokens"] == output_tokens)
            ].iloc[0]
            gm_rows.append({
                "machine": "gm700-desktop",
                "scenario": f"KV cache {cache}",
                "family": "4-token prompt",
                "prompt_tokens": 4,
                "output_tokens": int(output_tokens),
                "concurrency_or_batch": 1,
                "ttft_ms": row["ttft_p50_ms"],
                "tpot_ms": row["tpot_p50_ms"],
                "output_tps": row["output_tps_p50"],
            })
    batching = pd.read_csv(root / "batching/summary.csv")
    for _, row in batching.iterrows():
        gm_rows.append({
            "machine": "gm700-desktop",
            "scenario": f"Fixed batch {int(row['batch_size'])}",
            "family": "4-token prompt",
            "prompt_tokens": 4,
            "output_tokens": 32,
            "concurrency_or_batch": int(row["batch_size"]),
            "ttft_ms": None,
            "tpot_ms": None,
            "output_tps": row["output_tps"],
        })
    vllm_path = root / "vllm/summary.csv"
    if vllm_path.exists():
        vllm = pd.read_csv(vllm_path)
        for _, row in vllm.iterrows():
            gm_rows.append({
                "machine": "gm700-desktop",
                "scenario": f"vLLM c={int(row['concurrency'])}",
                "family": "128-token prompt",
                "prompt_tokens": 128,
                "output_tokens": 32,
                "concurrency_or_batch": int(row["concurrency"]),
                "ttft_ms": row["ttft_p50_ms"],
                "tpot_ms": row["tpot_p50_ms"],
                "output_tps": row["output_tps"],
            })

    gm = pd.DataFrame(gm_rows)
    laptop = laptop_summary_path
    laptop = pd.read_csv(laptop_summary_path)
    comparison = pd.concat(
        [
            laptop[
                [
                    "scenario",
                    "family",
                    "prompt_tokens",
                    "output_tokens",
                    "concurrency_or_batch",
                    "ttft_ms",
                    "tpot_ms",
                    "output_tps",
                ]
            ].assign(machine="rtx3050ti-laptop"),
            gm[
                [
                    "scenario",
                    "family",
                    "prompt_tokens",
                    "output_tokens",
                    "concurrency_or_batch",
                    "ttft_ms",
                    "tpot_ms",
                    "output_tps",
                ]
            ].assign(machine="gm700-desktop"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(root / "comparison/summary.csv", index=False)

    label = hardware_label(hardware)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for machine, group in comparison[
        comparison["scenario"].str.startswith("Fixed batch")
    ].groupby("machine"):
        axes[0].plot(
            group["concurrency_or_batch"],
            group["output_tps"],
            marker="o",
            label=machine,
        )
    axes[0].set_title("Fixed Batching Output Throughput")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Output tokens/s")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for machine, group in comparison[
        comparison["scenario"].str.startswith("vLLM c=")
    ].groupby("machine"):
        axes[1].plot(
            group["concurrency_or_batch"],
            group["ttft_ms"],
            marker="o",
            label=machine,
        )
    axes[1].set_title("vLLM TTFT vs. Concurrency")
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("RTX 3050 Ti vs GM700: GPT-2 Comparison")
    fig.text(
        0.5,
        0.01,
        f"{label}\nDirect 4-token prompt experiments are not equal to 128-token vLLM prompts.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, root / "comparison/unified_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    hardware = json.loads((root / "hardware.json").read_text(encoding="utf-8"))
    generate_baseline(root, hardware)
    generate_kv_cache(root, hardware)
    generate_batching(root, hardware)
    generate_vllm(root, hardware)
    generate_comparison(root, hardware)


if __name__ == "__main__":
    main()
