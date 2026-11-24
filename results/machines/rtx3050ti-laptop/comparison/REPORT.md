# RTX 3050 Ti GPT-2 Comparison

## Hardware
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4095.5 MiB, compute 8.6, driver 560.94
- AMD Ryzen 9 5900HS, 8C/16T, 15.41 GiB RAM
- Windows 11 + WSL2 Ubuntu 24.04.5

## Headline findings
- Fixed batching improved derived generation throughput from 55.61 tok/s at batch 1 to 422.91 tok/s at batch 8.
- vLLM throughput rose from 54.40 to 307.20 output tok/s as concurrency rose from 1 to 16, while median TTFT rose from 31.79 to 210.36 ms.
- The short KV-cache test was noisy: cache-on median E2E was 1049.44 ms versus 863.75 ms off, while median cache-on TTFT was lower. Do not generalize from three short 32-token runs.
- A stabilized scaling run clarified the tradeoff: at 128 output tokens,
  cache-on median E2E was 2423.47 ms versus 3328.55 ms off, a 27.19%
  reduction; median TPOT fell by 27.11%.

## Comparison
| Scenario | Prompt / output tokens | Median TTFT | Median TPOT | Output tok/s |
|---|---:|---:|---:|---:|
| PyTorch baseline | 4 / 32 | 29.21 | 25.27 | 38.78 |
| KV cache on | 4 / 32 | 19.84 | 32.94 | 30.49 |
| KV cache off | 4 / 32 | 30.53 | 26.60 | 37.05 |
| Fixed batch 1 | 4 / 32 | N/A | N/A | 55.61 |
| Fixed batch 2 | 4 / 32 | N/A | N/A | 98.71 |
| Fixed batch 4 | 4 / 32 | N/A | N/A | 190.92 |
| Fixed batch 8 | 4 / 32 | N/A | N/A | 422.91 |
| vLLM c=1 | 128 / 32 | 31.79 | 16.11 | 54.40 |
| vLLM c=2 | 128 / 32 | 52.65 | 25.03 | 82.67 |
| vLLM c=4 | 128 / 32 | 82.19 | 30.40 | 132.80 |
| vLLM c=8 | 128 / 32 | 124.58 | 35.86 | 217.86 |
| vLLM c=16 | 128 / 32 | 210.36 | 52.38 | 307.20 |

## Limitations
- vLLM used 128 prompt tokens; baseline/KV/batching used 4 prompt tokens.
- Throughput is comparable only as directional runtime behavior, not as a controlled runtime ranking.
- More repetitions, equal prompt lengths, and fixed clocks are required for a controlled final benchmark.
- The laptop reached 83-91 C during the KV-cache sweep and its graphics
  clock varied from 210-877 MHz, so thermal and power-state variance must
  be considered in cross-machine comparisons.
