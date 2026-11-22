import csv
import tempfile
from pathlib import Path

from src.benchmarks.metrics import percentile
from src.benchmarks.workload_generator import fixed_prompts
from src.benchmarks.metrics import append_benchmark_csv
from src.inference.minimal_decoder import GenerationMetrics
from src.inference.kv_cache import estimate_kv_cache_bytes


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
