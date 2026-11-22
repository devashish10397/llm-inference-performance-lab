import csv
import tempfile
from pathlib import Path

from src.benchmarks.metrics import percentile
from src.benchmarks.workload_generator import fixed_prompts
from src.benchmarks.metrics import append_benchmark_csv
from src.inference.minimal_decoder import GenerationMetrics
from src.inference.kv_cache import estimate_kv_cache_bytes
from src.serving.scheduler import FIFOScheduler, Request
from src.analysis.plot_results import plot_kv_cache, plot_batching


def test_percentile_interpolation() -> None:
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10], 95) == 10


def test_fixed_prompts() -> None:
    prompts = fixed_prompts(3, "benchmark prompt")
    assert prompts == ["benchmark prompt"] * 3


def test_fixed_prompts_rejects_zero() -> None:
    try:
        fixed_prompts(0)
    except ValueError:
        return
    raise AssertionError("fixed_prompts(0) should raise ValueError")


def test_append_benchmark_csv() -> None:
    metrics = GenerationMetrics(
        model="test-model",
        device="cpu",
        prompt="hello",
        prompt_tokens=1,
        generated_tokens=10,
        prefill_ms=20,
        ttft_ms=25,
        end_to_end_ms=100,
        tpot_ms=8,
        peak_gpu_memory_mb=None,
    )
    with tempfile.TemporaryDirectory() as directory:
        csv_path = append_benchmark_csv(
            Path(directory) / "runs.csv",
            metrics,
            repeat=2,
        )
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["repeat"] == "2"
    assert float(rows[0]["throughput_tokens_per_second"]) == 100


def test_estimate_kv_cache_bytes() -> None:
    class Config:
        n_layer = 12
        n_head = 12
        n_embd = 768
        num_key_value_heads = 12

    assert estimate_kv_cache_bytes(Config(), 128) == 12 * 128 * 12 * 64 * 4


def test_fifo_scheduler_batching() -> None:
    scheduler = FIFOScheduler(batch_size=3)
    scheduler.submit_many(
        Request(str(index), "prompt", 8)
        for index in range(7)
    )
    assert [len(batch) for batch in scheduler.drain()] == [3, 3, 1]


def test_kv_cache_and_batch_chart_helpers(tmp_path: Path) -> None:
    kv_csv = tmp_path / "kv.csv"
    kv_csv.write_text(
        "kv_cache_enabled,tpot_ms,ttft_ms,end_to_end_ms,"
        "peak_gpu_memory_mb\n"
        "True,6,7,50,253\n"
        "False,6,6,49,254\n",
        encoding="utf-8",
    )
    kv_charts = plot_kv_cache(kv_csv, tmp_path / "kv_charts")
    assert all(chart.exists() for chart in kv_charts)

    batch_csv = tmp_path / "batch.csv"
    batch_csv.write_text(
        "batch_size,batch_latency_ms,queue_wait_ms\n"
        "1,50,1\n"
        "2,60,2\n"
        "4,70,3\n",
        encoding="utf-8",
    )
    batch_charts = plot_batching(batch_csv, tmp_path / "batch_charts")
    assert all(chart.exists() for chart in batch_charts)
