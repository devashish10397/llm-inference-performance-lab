from src.benchmarks.metrics import percentile
from src.benchmarks.workload_generator import fixed_prompts


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
