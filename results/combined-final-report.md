# Laptop and Desktop LLM Inference Results

## Executive summary

The optimized RTX 3050 Ti laptop campaign improved several GPT-2 workloads:

- Fixed batching at batch 8 reached 240.57 output tokens/s, 534.15% above
  batch 1.
- Cache-on 128-output KV-cache generation reduced P50 end-to-end latency
  by 42.16% versus cache-off.
- vLLM concurrency 8 reached 137.06 output tokens/s, 183.96% above
  concurrency 1.
- Sustained execution exposed a thermal/power bottleneck: temperature
  stayed at 91-97 C, the graphics clock stayed at 210 MHz, and the P50
  end-to-end latency became 1167.65 ms.

The RTX 5060 Ti GM700 desktop remained substantially faster in the
direct and serving workloads:

- Direct PyTorch GPT-2: 167.64 output tokens/s.
- Fixed batch 8: 1253.88 output tokens/s.
- vLLM concurrency 16: 5539.43 output tokens/s.

The desktop's advantage is broad, but the 128-token vLLM workload is not
directly comparable to the 4-token direct PyTorch and batching workloads.

## Final optimized laptop table

| Experiment | Control | Optimized result | Improvement |
|---|---|---|---:|
| Fixed batching | Batch 1: 37.94 output tok/s | Batch 8: 240.57 output tok/s | +534.15% |
| KV cache, 32 output | Off: 971.09 ms E2E | On: 818.17 ms E2E | +15.75% |
| KV cache, 64 output | Off: 2198.71 ms E2E | On: 1645.28 ms E2E | +25.17% |
| KV cache, 128 output | Off: 5812.61 ms E2E | On: 3361.77 ms E2E | +42.16% |
| vLLM concurrency | C=1: 48.27 output tok/s | C=8: 137.06 output tok/s | +183.96% |
| Sustained execution | Short P50: 804.09 ms | Hot P50: 1167.65 ms | -45.21% |

## Laptop vs desktop throughput

| Scenario | RTX 3050 Ti | RTX 5060 Ti |
|---|---:|---:|
| Direct GPT-2, 4 prompt / 32 output | 39.80 output tok/s | 167.64 output tok/s |
| Fixed batch 8, 4 prompt / 32 output | 240.57 output tok/s | 1253.88 output tok/s |
| KV cache 128, cache-on, 4 prompt / 128 output | 38.08 output tok/s | 172.07 output tok/s |
| vLLM c=1, 128 prompt / 32 output | 48.27 output tok/s | 652.18 output tok/s |
| vLLM c=8, 128 prompt / 32 output | 137.06 output tok/s | 3768.27 output tok/s |

## Desktop-laptop gap

| Scenario | RTX 5060 Ti / RTX 3050 Ti throughput ratio |
|---|---:|
| Direct GPT-2 baseline | 4.21x |
| Fixed batch 8 | 5.21x |
| KV cache 128 on | 4.52x |
| vLLM c=1 | 13.51x |
| vLLM c=8 | 27.49x |

The vLLM ratios are not controlled runtime comparisons because the laptop
and desktop used different vLLM/PyTorch/CUDA stacks, and the desktop had
more VRAM and newer GPU architecture. They are useful as observed results,
not universal hardware scaling factors.

## Latency comparison

| Scenario | RTX 3050 Ti | RTX 5060 Ti |
|---|---:|---:|
| Direct TTFT P50 | 27.82 ms | 6.78 ms |
| Direct TPOT P50 | 24.63 ms | 5.85 ms |
| Direct E2E P50 | 804.09 ms | 190.89 ms |
| vLLM c=1 TTFT P50 | 43.74 ms | 11.30 ms |
| vLLM c=8 TTFT P50 | 287.72 ms | 24.27 ms |

The laptop's optimized batching and KV-cache improvements are meaningful,
but the short, cool desktop run still had lower TTFT and TPOT.

## Thermal and power

| Machine | Temperature range | Power range | Sustained state |
|---|---:|---:|---|
| RTX 3050 Ti Laptop | 91-97 C during sustained test | 19.16-24.10 W | 210 MHz graphics clock |
| RTX 5060 Ti Desktop | 37.18-44.18 C KV telemetry | 6.01-50.78 W | No sustained thermal test |

The laptop's thermal result is the clearest performance limitation:
after warming, the GPU remained at 210 MHz and the sustained P50 end-to-end
latency increased 45.21% relative to the short baseline.

## Conclusion

The RTX 3050 Ti laptop is usable for learning inference internals and
produces clear optimization signals. Fixed batching, KV-cache and vLLM
concurrency all improved laptop throughput and latency.

The laptop is still far behind the RTX 5060 Ti desktop in raw throughput:

- 4.21x lower direct GPT-2 throughput
- 5.21x lower fixed-batch throughput
- 27.49x lower vLLM c=8 throughput

However, these results show that a 4 GB laptop GPU is sufficient for:

- GPT-2 FP16 decoding
- KV-cache experiments
- Fixed batching
- Short-context vLLM serving
- Thermal and power-state observation
- Learning scheduling and serving concepts

It is not the right hardware for longer repeated 128+ concurrent workloads,
large models, stable high-throughput serving or quantized MLC benchmarking.

## MLC LLM status

MLC LLM was retried with the CUDA 12.8 wheel and native library paths.
GPU-visible Torch and MLC imports segfaulted, and CPU-only package imports
could not load all required TVM runtime libraries. No MLC model was downloaded
or benchmarked. This result should be documented as a compatibility limitation,
not as a runtime performance result.

## Reproduction

- [GM700 report](machines/gm700-desktop/README.md)
- [RTX 3050 Ti report](machines/rtx3050ti-laptop/README.md)
- [Laptop optimization report](machines/rtx3050ti-laptop/optimization/README.md)
- [Cross-machine summary](machines/gm700-desktop/comparison/summary.csv)
- [Optimization improvements](machines/rtx3050ti-laptop/optimization/improvements.csv)
