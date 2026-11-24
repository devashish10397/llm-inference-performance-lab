# RTX 3050 Ti Laptop Results

This directory is the compact, publishable result set for the 4 GB
NVIDIA GeForce RTX 3050 Ti laptop. Raw environments, downloaded models,
full GuideLLM JSON, temporary logs, and caches remain local.

## Environment

The exact hardware and software inventory is in
[hardware.json](hardware.json). Windows benchmarks used the repository
`.venv`; serving benchmarks used WSL2 with vLLM 0.8.5.post1 and GuideLLM
0.8.1.

## Workloads

| Experiment | Model | Input tokens | Output tokens | Repetitions |
|---|---|---:|---:|---|
| Baseline | GPT-2 | 4 | 32 | 1 warm-up, 3 measured |
| KV-cache scaling | GPT-2 | 4 | 32, 64, 128 | 1 warm-up, 5 measured per mode |
| Fixed batching | GPT-2 | 4 | 32 | 1 warm-up, 8 total requests at batch 1/2/4/8 |
| vLLM serving | GPT-2 | 128 | 32 | 120 seconds at concurrency 1/2/4/8/16 |

## Contents

- `baseline/`: direct PyTorch measurements and charts.
- `kv-cache/`: cache on/off scaling measurements, summary, telemetry plot,
  variance chart, and interpretation.
- `batching/`: fixed-batch FIFO measurements and charts.
- `vllm/`: compact serving metrics and charts.
- `comparison/`: unified directional comparison and workload caveats.

## Reproduction

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\inference\minimal_decoder.py --device cuda --warmup 1 --runs 3
.\.venv\Scripts\python.exe src\inference\kv_cache.py --device cuda --warmup 1 --runs 5 --max-new-tokens 32 --mode both
.\.venv\Scripts\python.exe src\inference\kv_cache.py --device cuda --warmup 1 --runs 5 --max-new-tokens 64 --mode both
.\.venv\Scripts\python.exe src\inference\kv_cache.py --device cuda --warmup 1 --runs 5 --max-new-tokens 128 --mode both
.\.venv\Scripts\python.exe src\serving\scheduler.py --device cuda --warmup 1 --requests 8 --batch-size 8
```

The existing local vLLM run used WSL2 with
`~/vllm-env/bin/vllm` and
`~/guidellm-env/bin/guidellm`. vLLM 0.8.5.post1 was selected
because its CUDA 12.4 PyTorch build is compatible with the laptop's
driver; vLLM 0.34 installed a CUDA 13 stack that the driver rejected.

## Comparison caution

The direct PyTorch experiments used a 4-token prompt, while the vLLM
serving benchmark used a 128-token prompt. Treat cross-runtime results as
directional until the same prompt length, fixed clocks, and identical
tokenization path are used on both the laptop and desktop.
