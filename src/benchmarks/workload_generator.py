"""Small deterministic workload definitions for baseline benchmarks."""

from __future__ import annotations


def fixed_prompts(count: int, prompt: str = "Large language model inference") -> list[str]:
    if count < 1:
        raise ValueError("count must be at least 1")
    return [prompt for _ in range(count)]