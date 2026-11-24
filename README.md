# LLM Inference Performance Lab

**Building, benchmarking, and optimizing LLM inference systems from first principles.**

A hands-on systems engineering project exploring the internals of Large Language Model (LLM) inference, with a focus on performance, scalability, memory efficiency, and GPU execution.

Rather than treating inference engines as black-box APIs, this project investigates how they work internally through implementation, profiling, and reproducible experiments.

> **Project status:** Initial development. Implementations, benchmarks, and experimental results will be added incrementally.

## Objectives

- **Inference internals:** Implement autoregressive token generation, prefill/decode execution, and KV-cache experiments.
- **Serving systems:** Explore request scheduling, continuous batching, concurrency, and resource management.
- **Inference runtimes:** Study vLLM for server-side inference and MLC LLM for compiled and hardware-aware inference.
- **Performance engineering:** Measure latency, throughput, memory usage, and CPU/GPU utilization.
- **Optimization:** Investigate performance bottlenecks, test improvements, and evaluate tradeoffs using reproducible workloads.

## System Architecture

The project is organized around four components:

| Component | Responsibility |
|---|---|
| Inference Engine | Minimal decoder, autoregressive generation, KV-cache experiments |
| Serving Layer | Request scheduling, batching, and serving experiments |
| Benchmarking Framework | Workload generation, concurrency testing, and metric collection |
| Performance Analysis | Profiling, visualizations, bottleneck analysis, and experiment comparisons |

The project will begin with a small PyTorch-based inference implementation and progressively introduce more advanced inference runtimes and serving frameworks.

## Technology Stack

| Area | Tools and Technologies |
|---|---|
| Programming | Python |
| Model Execution | PyTorch, Hugging Face Transformers |
| Server-side Inference | vLLM |
| Compiled / Local Inference | MLC LLM |
| GPU Performance | CUDA, NVIDIA profiling tools (where supported) |
| Data Analysis | NumPy, pandas, Matplotlib |
| Development | Git, GitHub, Windows, WSL2 |

Additional technologies will be introduced as experiments require them.

## Performance Metrics

The benchmarking framework will investigate:

| Metric | Description |
|---|---|
| Time to First Token (TTFT) | Time from request submission until the first generated token |
| Time per Output Token (TPOT) | Average generation time per output token after the first |
| End-to-End Latency | Total time required to complete an inference request |
| Request Throughput | Completed requests per second |
| Token Throughput | Generated tokens per second |
| Tail Latency | P50, P95, and P99 latency distributions |
| GPU Memory Usage | Model weights, KV-cache usage, and runtime memory consumption |
| GPU Utilization | GPU execution activity under different workloads |

Measurements will include workload configuration, hardware details, and methodology so that results can be interpreted and reproduced.

## Experiment Roadmap

| Phase | Focus | Status |
|---|---|---|
| 01 | Minimal autoregressive decoder and baseline measurements | Planned |
| 02 | KV-cache behavior and memory analysis | Planned |
| 03 | Request scheduling and batching experiments | Planned |
| 04 | vLLM serving, concurrency, and throughput analysis | Planned |
| 05 | MLC LLM runtime and quantization experiments | Planned |
| 06 | GPU profiling and bottleneck identification | Planned |
| 07 | Optimization experiments and performance comparisons | Planned |

## Repository Structure

```text
llm-inference-performance-lab/
├── configs/
├── src/
│   ├── inference/
│   │   ├── minimal_decoder.py
│   │   └── kv_cache.py
│   ├── serving/
│   │   └── scheduler.py
│   ├── benchmarks/
│   │   ├── workload_generator.py
│   │   └── metrics.py
│   └── analysis/
│       └── plot_results.py
├── experiments/
│   ├── 01_baseline/
│   ├── 02_kv_cache/
│   ├── 03_batching/
│   ├── 04_vllm/
│   └── 05_mlc_llm/
├── tests/
├── results/
├── docs/
├── requirements.txt
└── README.md
```

The initial source files are placeholders and will be implemented incrementally.

## Experimental Methodology

Each experiment will follow the same engineering process:

1. **Define:** Identify a performance question and formulate a hypothesis.
2. **Implement:** Build or modify the relevant inference component.
3. **Measure:** Collect baseline latency, throughput, and resource metrics.
4. **Analyze:** Identify bottlenecks and explain observed behavior.
5. **Optimize:** Introduce a controlled change and evaluate its impact.
6. **Document:** Record configurations, results, limitations, and conclusions.

No performance improvements will be claimed without supporting measurements.

## Getting Started

Create a Python 3.12 virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Run the minimal decoder baseline on NVIDIA CUDA:

```powershell
python src/inference/minimal_decoder.py --device cuda --max-new-tokens 32 --output results/baseline_smoke.json
```

Run a repeated baseline with warm-up:

```powershell
python src/inference/minimal_decoder.py --device cuda --warmup 1 --runs 5 --repeat 1 --max-new-tokens 32
```

Sweep prompt and output lengths:

```powershell
python src/inference/minimal_decoder.py --device cuda --warmup 1 --runs 3 --repeat 1 --prompt-token-counts "32,128,512" --output-token-counts "8,32,128"
```

Each run is also appended to `results/01_baseline/runs.csv` by default. Benchmark CSV
and chart files are local artifacts and are ignored by Git. Generate charts with:

```powershell
python src/analysis/plot_results.py
```

Generate charts for baseline, KV-cache, and batching results:

```powershell
python src/analysis/plot_results.py --all-experiments
```

Compare KV-cache generation with full-sequence recomputation:

```powershell
python src/inference/kv_cache.py --device cuda --warmup 1 --runs 3 --max-new-tokens 32 --mode both
```

Run a fixed-batch FIFO scheduling benchmark:

```powershell
python src/serving/scheduler.py --device cuda --warmup 1 --requests 8 --batch-size 4 --max-new-tokens 32
```

See [vLLM serving results](docs/04_vllm_results.md) for the completed concurrency experiment.

## Machine Benchmark Results

- [GM700 desktop](results/machines/gm700-desktop/README.md)
- [RTX 3050 Ti laptop](results/machines/rtx3050ti-laptop/README.md)

Development will initially use Windows and Python, with WSL2 and compatible GPU environments introduced for serving experiments.

## Learning and Engineering Focus

This project explores questions such as:

- Why can higher concurrency improve throughput while increasing tail latency?
- How does KV-cache memory grow with sequence length and active requests?
- When is inference compute-bound versus memory-bandwidth-bound?
- How does continuous batching affect GPU utilization?
- What are the performance tradeoffs of quantization?
- How do different inference runtimes behave under equivalent workloads?

## References

- [vLLM](https://github.com/vllm-project/vllm)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [GuideLLM](https://github.com/vllm-project/guidellm)

## Project Philosophy

**Build → Measure → Analyze → Optimize → Document**

The objective is not simply to run LLMs, but to understand the systems that execute them and the engineering tradeoffs that determine inference performance.