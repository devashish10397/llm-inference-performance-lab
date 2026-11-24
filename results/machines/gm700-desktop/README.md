# GM700 Desktop Results

This directory contains the publishable benchmark bundle for the GM700
desktop with an NVIDIA GeForce RTX 5060 Ti. It reproduces the methodology
used for the RTX 3050 Ti laptop bundle while keeping the two machines
separate.

## Environment

See [hardware.json](hardware.json) for the exact inventory.

- OS: Windows 11 Home, build 10.0.26300
- CPU: AMD Ryzen 7 8700F, 8 cores / 16 threads
- RAM: 31.67 GiB
- GPU: NVIDIA GeForce RTX 5060 Ti, 16311 MiB VRAM
- Compute capability: 12.0
- NVIDIA driver: 621.01
- Python: 3.12.10
- PyTorch: 2.11.0+cu128
- Transformers: 5.24.0
- vLLM: 0.34.0
- vLLM Torch stack: 2.14.0+cu130
- GuideLLM: 0.8.1
- Model: GPT-2 / FP16

## Direct PyTorch

Four prompt tokens and 32 output tokens, one warm-up and three measured runs.

| Metric | P50 |
|---|---:|
| TTFT | 6.7815 ms |
| TPOT | 5.8454 ms |
| End-to-end | 190.8867 ms |
| Output throughput | 167.6387 tok/s |
| Peak GPU memory | 254.0469 MiB |

## KV-Cache Scaling

Four prompt tokens, 32/64/128 output tokens, one warm-up and five measured
runs per cache mode.

| Output | Cache | TTFT P50 | TPOT P50 | E2E P50 | Output TPS P50 | Peak VRAM |
|---:|:---:|---:|---:|---:|---:|---:|
| 32 | On | 6.8153 ms | 5.7610 ms | 187.4828 ms | 170.6823 | 254.0469 MiB |
| 32 | Off | 6.1225 ms | 5.7734 ms | 186.9556 ms | 171.1636 | 259.2896 MiB |
| 64 | On | 6.8056 ms | 5.7958 ms | 376.3076 ms | 170.0736 | 255.1978 MiB |
| 64 | Off | 6.1481 ms | 6.0441 ms | 391.5349 ms | 163.4592 | 265.4878 MiB |
| 128 | On | 6.7712 ms | 5.7357 ms | 743.8890 ms | 172.0687 | 257.5732 MiB |
| 128 | Off | 5.9978 ms | 6.1404 ms | 795.8597 ms | 160.8324 | 277.8838 MiB |

At 128 output tokens, cache-on median end-to-end latency was 52.0 ms lower
than cache-off (6.54% lower), and TPOT was 0.4047 ms lower (6.59% lower).

## GPU Telemetry

Telemetry was captured once per second during the KV-cache sweep.

| Metric | Range |
|---|---:|
| Temperature | 37.18-44.18 C |
| Power | 6.01-50.78 W |
| GPU utilization | 0-44% |
| Memory utilization | 0-49% |
| Graphics clock | 510-2775 MHz |
| Memory clock | 405-14001 MHz |
| VRAM used | 1133-1628 MiB |

## Fixed Batching

Eight GPT-2 requests, 32 output tokens, one warm-up.

| Batch size | Output TPS | Batch latency P50 | Queue wait P50 |
|---:|---:|---:|---:|
| 1 | 169.9085 | 186.7834 ms | 669.0342 ms |
| 2 | 308.2278 | 204.8480 ms | 321.1365 ms |
| 4 | 638.8834 | 200.3495 ms | 101.9945 ms |
| 8 | 1253.8829 | 204.1658 ms | 0.0461 ms |

## vLLM Serving

GPT-2 completion endpoint, 128 prompt tokens and 32 output tokens, 120 seconds
per concurrency level. All five levels recorded zero errored requests.

| Concurrency | TTFT P50 | TTFT P95 | TPOT P50 | Output TPS | Requests/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.2984 ms | 13.1691 ms | 1.5028 ms | 652.1833 | 20.375 |
| 2 | 17.2665 ms | 19.1526 ms | 1.7432 ms | 1171.5639 | 36.600 |
| 4 | 19.5813 ms | 21.4849 ms | 1.8899 ms | 2160.1083 | 67.483 |
| 8 | 24.2724 ms | 26.5276 ms | 2.1617 ms | 3768.2667 | 117.758 |
| 16 | 39.7477 ms | 43.4449 ms | 2.9429 ms | 5539.4333 | 173.067 |

## Contents

- `baseline/`: direct PyTorch raw and summary measurements and charts.
- `kv-cache/`: cache scaling, variance, summary and one-second telemetry.
- `batching/`: fixed-batch measurements and charts.
- `vllm/`: completion scenario, normalized summary and charts.
- `comparison/`: matched cross-machine normalized summary and chart.

## Limitations

- Direct PyTorch/KV/batching used a 4-token prompt.
- vLLM used a 128-token prompt.
- These prompt lengths are not directly equivalent; use the comparison only
  to observe directional differences.
- vLLM and direct PyTorch used different serving and timing paths.
- Raw GuideLLM JSON and long logs are not included in this publishable bundle.
