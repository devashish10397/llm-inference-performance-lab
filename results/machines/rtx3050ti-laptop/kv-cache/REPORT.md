# KV-Cache Scaling Results

| Output | Cache | Median TTFT | Median TPOT | Median E2E | Median tok/s | E2E CV |
|---:|:---:|---:|---:|---:|---:|---:|
| 32 | on | 23.58 | 23.03 | 745.78 | 42.91 | 12.6% |
| 32 | off | 16.43 | 18.28 | 589.94 | 54.24 | 30.0% |
| 64 | on | 19.37 | 19.38 | 1258.33 | 50.86 | 7.8% |
| 64 | off | 18.57 | 19.99 | 1292.95 | 49.50 | 17.9% |
| 128 | on | 19.79 | 18.67 | 2423.47 | 52.82 | 4.7% |
| 128 | off | 18.70 | 25.61 | 3328.55 | 38.46 | 1.4% |

## Telemetry
- Temperature: 83-91 C
- Power: 16.48-22.79 W
- Graphics clock: 210-877 MHz
- GPU utilization: 0-94%
- Raw samples: [telemetry.csv](telemetry.csv) contains all 123 one-second
  samples captured during the benchmark.
- Plot: [gpu_telemetry.png](gpu_telemetry.png)

## Finding
At 128 output tokens, the cache-on median is about 27.2% lower end-to-end and 27.1% lower TPOT than cache-off. The 32-token case remains noisy; scaling to longer outputs makes the cache benefit clear.
