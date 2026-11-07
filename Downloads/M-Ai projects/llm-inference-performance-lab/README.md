# LLM Inference Performance Lab

A hands-on systems engineering project focused on understanding, implementing, benchmarking, and optimizing Large Language Model (LLM) inference systems.

The goal is to explore how inference engines work internally, identify performance bottlenecks, and evaluate optimization techniques through reproducible experiments.

**Status:** Work in progress — repository setup.

## Project Objectives

- Build a minimal autoregressive inference engine using PyTorch.
- Understand and implement KV caching and token generation.
- Explore request scheduling, batching, and concurrency.
- Study vLLM's serving architecture and performance characteristics.
- Explore MLC LLM, model compilation, quantization, and inference runtimes.
- Develop benchmarking tools for latency, throughput, and resource utilization.
- Profile inference workloads and investigate CPU/GPU performance bottlenecks.
- Document experimental findings and engineering tradeoffs.

## Architecture (Planned)

The project will evolve through four components:

1. **Inference Engine:** Minimal decoder, token generation, and KV-cache experiments.
2. **Serving and Scheduling:** Request scheduling, batching, and inference serving experiments.
3. **Benchmarking Framework:** Workload generation, latency measurements, throughput analysis, and reproducible benchmarks.
4. **Performance Analysis:** CPU/GPU profiling, visualization, bottleneck identification, and optimization comparisons.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Inference Serving | vLLM |
| Inference Runtime | MLC LLM |
| GPU Performance | CUDA, NVIDIA profiling tools (where supported) |
| Benchmarking | Python, NumPy, pandas |
| Visualization | Matplotlib |
| Development | Git, GitHub, Windows, WSL2 |

Technologies will be introduced incrementally as the project develops.

## Performance Metrics

The experiments will focus on:

- **TTFT:** Time to first token
- **TPOT:** Time per output token
- **End-to-end latency:** Total request completion time
- **Throughput:** Requests/sec and tokens/sec
- **Tail latency:** P50, P95, and P99
- **GPU utilization and memory consumption**
- **Concurrency and batching efficiency**

## Experiment Roadmap

| Phase | Experiment | Status |
|---|---|---|
| 1 | Minimal autoregressive decoder | Planned |
| 2 | KV-cache implementation and comparison | Planned |
| 3 | Request scheduling and batching | Planned |
| 4 | vLLM serving and benchmarking | Planned |
| 5 | MLC LLM runtime experiments | Planned |
| 6 | GPU profiling and bottleneck analysis | Planned |
| 7 | Optimization experiments and comparisons | Planned |

## Repository Structure

```text
configs/         Experiment configurations
src/
  inference/     Minimal decoder and KV-cache experiments
  serving/       Scheduling and batching
  benchmarks/    Workload generation and performance metrics
  analysis/      Data analysis and visualization
experiments/     Reproducible experiments
tests/           Automated tests
results/         Generated benchmark outputs
docs/            Architecture notes and findings
```

## Experimental Methodology

Each experiment will follow a consistent process:

1. Define the performance question or hypothesis.
2. Establish a reproducible baseline.
3. Collect latency, throughput, and resource metrics.
4. Introduce a controlled change.
5. Compare results under equivalent workloads.
6. Document observations, limitations, and conclusions.

Performance claims will be supported by measured results.

## Current Progress

- [x] Initialize GitHub repository
- [x] Create initial project structure
- [ ] Implement minimal inference decoder
- [ ] Capture baseline inference measurements
- [ ] Compare inference with and without KV caching
- [ ] Build a concurrent workload generator
- [ ] Run vLLM serving experiments

## References

- [vLLM](https://github.com/vllm-project/vllm)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Project Philosophy

**Build → Measure → Analyze → Optimize → Document**

The focus is on understanding inference systems through implementation and empirical performance engineering rather than treating inference engines as black-box APIs.