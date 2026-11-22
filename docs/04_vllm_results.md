# vLLM Serving Results

## Environment

- GPU: NVIDIA GeForce RTX 5060 Ti
- GPU memory: 16 GB
- Runtime: vLLM 0.34.0
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Precision: FP16
- Platform: Windows 11 with WSL2 Ubuntu
- Workload: 128 prompt tokens and 32 output tokens
- Duration: 120 seconds per concurrency level

## Concurrency Results

| Concurrency | TTFT median | TPOT median | Output TPS | Requests/s |
|---:|---:|---:|---:|---:|
| 1 | 11.02 ms | 3.86 ms | 101.07 | 3.16 |
| 2 | 15.95 ms | 3.73 ms | 232.26 | 7.25 |
| 4 | 22.70 ms | 3.96 ms | 446.67 | 13.96 |
| 8 | 40.45 ms | 4.57 ms | 746.67 | 23.33 |
| 16 | 61.66 ms | 5.32 ms | 1049.32 | 32.78 |

## Observations

- Throughput increased from 101 to 1049 output tokens/s as concurrency increased from 1 to 16.
- Median TTFT increased from 11.02 ms to 61.66 ms over the same range.
- TPOT remained relatively stable between 3.73 ms and 5.32 ms.
- The observed throughput gains came with a clear TTFT latency tradeoff.

## Reproduction

```powershell
wsl -d Ubuntu-24.04
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --port 8000
```

In another Ubuntu terminal:

```bash
source ~/guidellm-env/bin/activate
guidellm run --scenario ~/guidellm-smoke.json --disable-console-interactive
```
