# GM700 vs. RTX 3050 Ti Laptop Comparison

## Hardware

| Machine | GPU | VRAM | CPU | RAM | OS | NVIDIA driver |
|---|---|---:|---|---:|---|---:|
| GM700 desktop | RTX 5060 Ti | 16311 MiB | Ryzen 7 8700F | 31.67 GiB | Windows 11 | 621.01 |
| RTX 3050 Ti laptop | RTX 3050 Ti Laptop | 4095.5 MiB | Ryzen 9 5900HS | 15.41 GiB | Windows 11 + WSL2 | 560.94 |

## Workload equality

Equal dimensions:
- Model: GPT-2
- Direct PyTorch precision: FP16
- Direct output length: 32 tokens
- KV-cache output lengths: 32, 64 and 128 tokens
- Fixed batching: 8 requests, batches 1/2/4/8
- vLLM model and 32-output-token workload

Unequal dimensions:
- Direct PyTorch/KV/batching used a 4-token prompt.
- vLLM used a 128-token prompt.
- Systems used different GPUs, VRAM, CPU, RAM and driver stacks.
- vLLM used different runtime versions and environments.

Do not rank prompt-sensitive or runtime-sensitive experiments as controlled
cross-machine comparisons unless the prompt length and server stack are equal.

## Headline comparisons

| Scenario | RTX 3050 Ti | GM700 desktop |
|---|---:|---:|
| PyTorch baseline TTFT P50 | 29.212 ms | 6.7815 ms |
| PyTorch baseline TPOT P50 | 25.274 ms | 5.8454 ms |
| PyTorch baseline output TPS | 38.782 | 167.6387 |
| Fixed batch 8 output TPS | 422.908 | 1253.8829 |
| KV cache 128 on/off E2E P50 | 2423.47 / 3328.55 ms | 743.889 / 795.860 ms |
| vLLM c=1 TTFT P50 | 31.791 ms | 11.2984 ms |
| vLLM c=16 TTFT P50 | 210.357 ms | 39.7477 ms |
| vLLM c=1 output TPS | 54.40 | 652.1833 |
| vLLM c=16 output TPS | 307.20 | 5539.4333 |

## Interpretation

The GM700 desktop is substantially faster in the measured GPT-2 baseline,
fixed batching and vLLM serving workloads. Its 16 GB VRAM and newer GPU
architecture support larger KV-cache, batching and serving footprints.

The KV-cache 128-token result shows cache-on reduced median GM700 E2E from
795.8597 ms to 743.8890 ms. This is a 6.54% E2E reduction and a 6.59%
TPOT reduction.

The vLLM results show throughput scaling from 652.18 output tokens/s at
concurrency 1 to 5539.43 at concurrency 16. Median TTFT rose from 11.2984 ms
to 39.7477 ms over the same range.
