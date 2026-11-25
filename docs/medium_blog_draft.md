# Do You Need a Good GPU to Learn LLM Inference and Host Models Locally?

I started wondering about this after looking at the computer already on my
desk: an ASUS ROG Flow X13, an ultra-portable gaming laptop with an RTX 3050 Ti.
It was not a server. It was not a workstation. It was the machine I used for
normal work and for quick, local experiments.

This question brought back memories from my master's degree at Carnegie Mellon
University in 2023. I was working on applied deep-learning projects involving
emotion detection and transformer-based methods. Training was an exercise in
adjustment: choose a smaller model, reduce the batch size, fit the data into
GPU memory, retry, and repeat. Those constraints taught me useful engineering
habits, but they also made me wonder whether I needed the same sacrifice just
to understand modern LLM inference.

Today, transformer models are much easier to find and run. The harder question
is no longer only whether a model can run. It is whether it can run efficiently:

- How long does the first token take?
- How fast is every output token?
- How much GPU memory does the KV cache consume?
- Does batching improve throughput?
- How does concurrency affect tail latency?
- Does the GPU throttle under a sustained workload?

I did not want to learn only by reading about these concepts. I wanted to
measure them.

## The laptop, the cloud, and the desktop

Before buying another GPU, I considered the alternatives.

The ASUS ROG Flow X13 was already available. It was small, portable and had a
dedicated RTX 3050 Ti. An NVIDIA GPU is more useful for learning than a CPU
alone, but I did not know whether 4 GB of VRAM would be enough for serving and
inference experiments.

A cloud GPU VM was another option. It had one major advantage: I could use a
newer GPU and pay only for the time I needed. It had two equally real
drawbacks:

1. GPU VM quotas are not always easy to obtain, even when a cloud account is
   ready to pay.
2. The hourly cost grows quickly during repeated experiments.

GPU availability varies by region and VM family. A request for quota is not
always immediate, and an expensive machine may remain idle while waiting for
the right setup or account approval. A local GPU was less flexible, but it
gave me unlimited sessions as long as the hardware was available.

Then a good Black Friday deal appeared for a GM700 desktop. It has an AMD
Ryzen 7 8700F, 32 GB of RAM and an NVIDIA GeForce RTX 5060 Ti with 16 GB of
VRAM.

It was not only a purchase for LLM experiments. I also wanted the gaming PC I
had dreamed about as a child, with GTA 6 on the horizon for PC. It felt worth
buying when it could serve both purposes.

Before choosing a side, I ran the same small GPT-2 workload on both machines.

## What I measured

I used GPT-2 as a small baseline model. It was not intended to represent a
large production LLM. It was large enough to expose inference behavior while
remaining safe for the RTX 3050 Ti's 4 GB VRAM.

The workloads included:

- A direct PyTorch FP16 decoder
- KV-cache on/off decoding
- Fixed request batching
- vLLM continuous-style serving
- A five-minute sustained test

The direct tests used 4 prompt tokens and 32 generated tokens unless stated
otherwise. The vLLM tests used 128 prompt tokens and 32 generated tokens.
This difference is important: direct 4-token experiments are not identical to
128-token serving tests.

I recorded TTFT, TPOT, end-to-end latency, throughput, P50/P95/P99 tail
latency, peak VRAM and GPU telemetry.

## The baseline

The first comparison used GPT-2, FP16, one warm-up and three direct measured
runs.

| Machine | TTFT P50 | TPOT P50 | E2E P50 | Output TPS |
|---|---:|---:|---:|---:|
| RTX 3050 Ti laptop | 29.21 ms | 25.27 ms | 825.12 ms | 38.78 |
| RTX 5060 Ti desktop | 6.78 ms | 5.85 ms | 190.89 ms | 167.64 |

The desktop produced roughly 4.21 times the baseline output throughput. It
also cut TTFT to less than a quarter of the laptop result.

That was expected, but it was only the first experiment.

## The KV cache matters

KV caching is not a small implementation detail. It determines whether decode
steps reuse previous keys and values or recompute more context.

I tested cache-on and cache-off at 32, 64 and 128 output tokens.

| Output | RTX 3050 Ti cache-on | RTX 3050 Ti cache-off |
|---:|---:|---:|
| 32 | 818.17 ms E2E | 971.09 ms E2E |
| 64 | 1645.28 ms E2E | 2198.71 ms E2E |
| 128 | 3361.77 ms E2E | 5812.61 ms E2E |

The cache-on desktop result at 128 output tokens was 743.89 ms E2E. Cache-off
was 795.86 ms. That is a smaller gain than on the laptop because the desktop
was already faster, but the result still shows that KV cache changes the
decode cost.

At 128 output tokens on the laptop:

- Cache-on TPOT: 25.52 ms
- Cache-off TPOT: 44.27 ms
- Cache-on E2E: 3361.77 ms
- Cache-off E2E: 5812.61 ms

The cache-on laptop version completed 128-token generation in 42.16% less
time.

## Fixed batching improved throughput

The batch benchmark used eight requests and 32 output tokens for each batch
size.

| Batch size | Laptop output TPS | Desktop output TPS |
|---:|---:|---:|
| 1 | 37.94 | 169.91 |
| 2 | 68.13 | 308.23 |
| 4 | 122.23 | 638.88 |
| 8 | 240.57 | 1253.88 |

The laptop batch-8 result was 534.15% above batch 1. Batch size is therefore
one of the clearest ways to improve throughput on limited hardware.

The desktop improved even more, reaching 1253.88 output tokens/s at batch 8.
It showed the value of larger VRAM and stronger GPU compute without changing
the model or workload.

## vLLM increased throughput, but latency followed too

The vLLM serving workload used GPT-2 with 128 prompt tokens and 32 output
tokens. It ran for 120 seconds at concurrency 1, 2, 4, 8 and 16.

| Concurrency | Laptop TTFT P50 | Laptop output TPS | Desktop TTFT P50 | Desktop output TPS |
|---:|---:|---:|---:|---:|
| 1 | 43.74 ms | 48.27 | 11.30 ms | 652.18 |
| 2 | 77.25 ms | 70.40 | 17.27 ms | 1171.56 |
| 4 | 175.83 ms | 104.53 | 19.58 ms | 2160.11 |
| 8 | 287.72 ms | 137.06 | 24.27 ms | 3768.27 |
| 16 | Not captured | Not captured | 39.75 ms | 5539.43 |

Both systems followed the expected pattern:

- Throughput increased with concurrency.
- TTFT increased with concurrency.
- Tail latency became more sensitive to queueing and GPU saturation.

The laptop and desktop runtime versions were not identical. The laptop
used vLLM 0.8.5 with PyTorch 2.6/CUDA 12.4, while the desktop used vLLM 0.34
with PyTorch 2.14/CUDA 13.0. These are useful observations, not a universal
hardware scaling formula.

## The laptop thermal limit was real

The final laptop experiment ran cache-on GPT-2 generation for five minutes.
It completed 261 requests.

- Temperature: 91-97 C
- Power: 19.16-24.10 W
- Graphics clock: fixed at 210 MHz
- P50/P95/P99 end-to-end latency: 1167.65 / 1593.89 / 2169.09 ms
- Peak measured VRAM use: 907 MiB

The sustained P50 end-to-end latency was 45.21% worse than the short baseline.
The clock did not move above 210 MHz during the entire telemetry capture.

This is not proof that every laptop GPU performs this way. It does show that
long-running inference tests can become thermal/power constrained, and the
short benchmark may hide that problem.

## Does the laptop still close the gap?

Optimization made the laptop materially faster:

- Batch 8 was 6.34 times faster than batch 1.
- KV cache-on cut 128-token E2E by 42.16%.
- vLLM concurrency 8 was 183.96% faster than concurrency 1.

But the desktop still won the raw throughput comparison:

- Direct GPT-2: 167.64 versus 39.80 output tokens/s
- Fixed batch 8: 1253.88 versus 240.57 output tokens/s
- vLLM concurrency 8: 3768.27 versus 137.06 output tokens/s

As ratios, the desktop delivered:

- 4.21 times direct throughput
- 5.21 times batch-8 throughput
- 27.49 times vLLM c=8 throughput

The vLLM ratio is especially sensitive to runtime versions and hardware, so
it should not be treated as a permanent 27x hardware multiplier.

## What did I learn?

An old laptop GPU is sufficient to learn the major LLM inference concepts:

- Prefill and decode execution
- KV-cache memory
- Fixed and continuous batching
- TTFT and TPOT
- Tail latency
- Throughput versus concurrency
- GPU memory and clock behavior
- Thermal throttling

It is not enough for every goal. A desktop GPU is more appropriate for:

- Larger models
- Longer contexts
- High concurrency
- Stable long runs
- Larger batches
- Frequent serving tests
- Quantized model comparisons

MLC LLM was also considered, but I could not establish a stable RTX 5060 Ti
CUDA 12.8/13.0 setup in WSL2. No MLC model or MLC GPU benchmark is included
in this article.

## Should you buy a GPU?

If the goal is to learn, start with the GPU already available.

The RTX 3050 Ti laptop could run GPT-2, measure KV-cache behavior, demonstrate
batch throughput and serve an OpenAI-compatible vLLM endpoint. It is enough to
learn the fundamentals and discover why larger GPUs help.

If the goal is scale, repeated serving, larger models or fewer surprises, the
desktop GPU was worth it. The RTX 5060 Ti cut latency sharply and made high
concurrency practical with 16 GB of VRAM.

My practical conclusion is simple:

> **Buy a GPU when the experiments stop being limited by understanding and
> start being limited by hardware.**

## Reproducibility

All benchmark code and publishable results are available in the
[LLM Inference Performance Lab](https://github.com/devashish10397/llm-inference-performance-lab).

- [GM700 desktop report](https://github.com/devashish10397/llm-inference-performance-lab/tree/master/results/machines/gm700-desktop)
- [RTX 3050 Ti laptop report](https://github.com/devashish10397/llm-inference-performance-lab/tree/master/results/machines/rtx3050ti-laptop)
- [Laptop optimization report](https://github.com/devashish10397/llm-inference-performance-lab/tree/master/results/machines/rtx3050ti-laptop/optimization)
- [Combined final report](https://github.com/devashish10397/llm-inference-performance-lab/tree/master/results/combined-final-report.md)
