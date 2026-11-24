# ASUS RTX 3050 Ti GPT-2 Optimization

## Goal

Measure GPT-2 FP16 optimization levers on the ASUS laptop without changing
the existing benchmark source or the published GM700 desktop results.

## Hardware and software

See [hardware.json](hardware.json) for the exact inventory.

- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4095.5 MiB, compute capability 8.6
- AMD Ryzen 9 5900HS, 8 cores / 16 threads, 15.41 GiB RAM
- NVIDIA driver 560.94, driver CUDA capability 12.6
- Windows Python 3.12.2, PyTorch 2.14.1+cu126
- WSL2 Ubuntu 24.04.5, vLLM 0.8.5.post1, PyTorch 2.6.0+cu124
- GuideLLM 0.8.1

## Controlled sequence

| Phase | Fixed controls | One-variable change |
|---|---|---|
| A. Baseline | GPT-2, FP16, 4 prompt / 32 output, KV on, 1 warm-up | Five independent measured runs |
| B. Batching | GPT-2, FP16, 8 requests, 32 output, 1 warm-up | Batch size 1, 2, 4, 8 |
| C. KV scaling | GPT-2, FP16, 4 prompt, 1 warm-up, 5 runs per mode | Output length 32/64/128 and cache on/off |
| D. vLLM | GPT-2, 128 prompt / 32 output, 120 seconds, FP16 | Concurrency 1, 2, 4, 8 |
| E. Thermal | GPT-2, FP16, KV on, 4 prompt / 32 output | Elapsed time for five minutes |

No model caches, WSL environments, full GuideLLM JSON, logs, or secrets are
included in this directory.

## Baseline

Five measured runs:

- TTFT P50/P95/P99: 27.82 / 28.21 / 28.23 ms
- TPOT P50/P95/P99: 24.63 / 28.43 / 29.18 ms
- End-to-end P50/P95/P99: 804.09 / 924.56 / 948.55 ms
- Output throughput P50: 39.80 tokens/s
- Peak GPU allocation: 253.05 MiB

## Fixed batching

| Batch | Output tokens/s | P50 latency | P95 latency | P99 latency |
|---:|---:|---:|---:|---:|
| 1 | 37.94 | 829.52 ms | 923.16 ms | 954.33 ms |
| 2 | 68.13 | 906.18 ms | 1040.48 ms | 1057.58 ms |
| 4 | 122.23 | 1047.22 ms | 1109.94 ms | 1115.52 ms |
| 8 | 240.57 | 1064.12 ms | 1064.12 ms | 1064.12 ms |

Batch 8 delivered about 6.34x the batch-1 output throughput while reducing
median queue wait from 3011.56 ms to 0.10 ms.

## KV-cache scaling

| Output | Cache | TTFT P50 | TPOT P50 | E2E P50 | Output tokens/s |
|---:|:---:|---:|---:|---:|---:|
| 32 | On | 28.34 ms | 25.05 ms | 818.17 ms | 39.11 |
| 32 | Off | 27.40 ms | 29.06 ms | 971.09 ms | 32.95 |
| 64 | On | 27.98 ms | 24.92 ms | 1645.28 ms | 38.90 |
| 64 | Off | 26.96 ms | 33.10 ms | 2198.71 ms | 29.11 |
| 128 | On | 28.75 ms | 25.52 ms | 3361.77 ms | 38.08 |
| 128 | Off | 28.19 ms | 44.27 ms | 5812.61 ms | 22.02 |

At 128 output tokens, KV cache-on reduced median end-to-end latency by
about 42.2% and TPOT by about 42.4%.

## vLLM serving

All four accepted levels had zero errored requests.

| Concurrency | Successful | Errors | TTFT P50 | TTFT P95 | Output tokens/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 181 | 0 | 43.74 ms | 52.97 ms | 48.27 |
| 2 | 264 | 0 | 77.25 ms | 102.69 ms | 70.40 |
| 4 | 392 | 0 | 175.83 ms | 259.60 ms | 104.53 |
| 8 | 514 | 0 | 287.72 ms | 453.08 ms | 137.06 |

## Sustained thermal result

The five-minute KV-cache-on test completed 261 requests.

- End-to-end latency P50/P95/P99: 1167.65 / 1593.89 / 2169.09 ms
- Temperature: 91-97 C
- Power: 19.16-24.10 W
- Graphics clock: 210 MHz in all 363 samples
- Peak observed system VRAM use: 907 MiB

A cold-to-hot clock transition was not captured, but the GPU was in a
sustained 91-97 C, 210 MHz state throughout. This strongly indicates a
thermal/power-constrained condition and makes absolute latency unsuitable
for comparison with a cool-start run.

## Improvement comparison

The complete machine-readable control-versus-candidate table is in
[improvements.csv](improvements.csv). Positive `improvement_pct` always
means a better result; TTFT/TPOT/latency use a lower-is-better sign, while
throughput uses a higher-is-better sign.

| Change | Control | Candidate | Improvement |
|---|---:|---:|---:|
| Fixed batching | Batch 1: 37.94 tok/s | Batch 8: 240.57 tok/s | +534.15% |
| KV cache, 32 output | Off: 971.09 ms E2E | On: 818.17 ms E2E | +15.75% |
| KV cache, 64 output | Off: 2198.71 ms E2E | On: 1645.28 ms E2E | +25.17% |
| KV cache, 128 output | Off: 5812.61 ms E2E | On: 3361.77 ms E2E | +42.16% |
| vLLM concurrency | C=1: 48.27 tok/s | C=8: 137.06 tok/s | +183.96% |
| Five-minute sustain | Short baseline P50: 804.09 ms | Sustained P50: 1167.65 ms | -45.21% |

The final row is a degradation, not an optimization claim: it quantifies
the observed hot-state penalty during sustained execution.

## Artifacts

- `baseline/`: raw measurements, normalized summary, latency chart
- `batching/`: raw measurements, normalized percentiles, chart
- `kv-cache/`: raw measurements, normalized summary, scaling chart
- `vllm/`: scenario and normalized zero-error summary, serving chart
- `thermal/`: request latency, normalized one-second telemetry, summary,
  sustained test chart

## Limitations and failed/not-attempted configurations

- The campaign was run sequentially on one thermally constrained laptop;
  later phases inherited heat from earlier work.
- The sustained test began after the GPU was already hot, so a cold-start
  transition cannot be measured from this dataset.
- The direct PyTorch and vLLM prompt lengths differ (4 versus 128 tokens);
  compare each workload to its own control, not as an identical-runtime test.
- No OOM or driver failure occurred during this optimization campaign.
- No MLC LLM or model larger than GPT-2 was downloaded or run.
- vLLM 0.34 with its CUDA 13 stack was not attempted in this campaign; the
  existing, already-verified vLLM 0.8.5.post1 CUDA 12.4 environment was used
  because it is compatible with driver 560.94.
- No repository source file was changed.
