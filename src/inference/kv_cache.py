"""KV-cache versus recomputation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.metrics import append_benchmark_csv, percentile


def estimate_kv_cache_bytes(
    config: object,
    sequence_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Estimate K and V cache bytes for a transformer using K * V storage."""
    if sequence_length < 0:
        raise ValueError("sequence_length must not be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    n_layer = int(getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)))
    n_head = int(getattr(config, "n_head", getattr(config, "num_attention_heads", 1)))
    n_kv_head = int(
        getattr(config, "num_key_value_heads", n_head)
        if getattr(config, "num_key_value_heads", None) is not None
        else n_head
    )
    n_embd = int(
        getattr(config, "n_embd", getattr(config, "hidden_size", n_head))
    )
    head_size = n_embd // n_head
    return 2 * batch_size * sequence_length * n_layer * n_kv_head * head_size * dtype_bytes


@dataclass
class KVCacheMetrics:
    model: str
    device: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    ttft_ms: float
    end_to_end_ms: float
    tpot_ms: float
    peak_gpu_memory_mb: float | None
    max_new_tokens: int
    kv_cache_enabled: bool
    estimated_kv_cache_bytes: int
    gpu_name: str | None = None
    torch_version: str = torch.__version__


class KVCacheBenchmark:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        use_cache: bool,
    ) -> tuple[str, KVCacheMetrics]:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_tokens = input_ids.shape[1]

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        decode_times: list[float] = []
        past_key_values = None
        generated_ids: list[torch.Tensor] = []

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - start) * 1000

            past_key_values = outputs.past_key_values if use_cache else None
            first_token_start = time.perf_counter()
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if self.device == "cuda":
                torch.cuda.synchronize()
            ttft_ms = prefill_ms + (time.perf_counter() - first_token_start) * 1000
            generated_ids.append(next_token)

            for _ in range(max_new_tokens - 1):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                step_start = time.perf_counter()
                if use_cache:
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=self.device,
                            ),
                        ),
                        dim=1,
                    )
                    outputs = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    full_ids = torch.cat(
                        [input_ids, *[token for token in generated_ids]],
                        dim=1,
                    )
                    full_mask = torch.ones_like(full_ids)
                    outputs = self.model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        use_cache=False,
                    )

                if self.device == "cuda":
                    torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - step_start) * 1000)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token)

        end_to_end_ms = (time.perf_counter() - start) * 1000
        generated = torch.cat(generated_ids, dim=-1).to("cpu")
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        tpot_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if self.device == "cuda"
            else None
        )
        context_length = prompt_tokens + len(generated_ids)
        cache_bytes = estimate_kv_cache_bytes(
            self.model.config,
            context_length,
            dtype_bytes=2 if self.device == "cuda" else 4,
        )

        metrics = KVCacheMetrics(
            model=self.model_name,
            device=self.device,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            prefill_ms=prefill_ms,
            ttft_ms=ttft_ms,
            end_to_end_ms=end_to_end_ms,
            tpot_ms=tpot_ms,
            peak_gpu_memory_mb=peak_memory,
            max_new_tokens=max_new_tokens,
            kv_cache_enabled=use_cache,
            estimated_kv_cache_bytes=cache_bytes,
            gpu_name=(
                torch.cuda.get_device_name(0)
                if self.device == "cuda"
                else None
            ),
        )
        return text, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KV cache with recomputation")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--prompt", default="Large language model inference")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=("both", "on", "off"), default="both")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/02_kv_cache/runs.csv"),
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    benchmark = KVCacheBenchmark(args.model, args.device)
    modes = (True, False) if args.mode == "both" else ((True,) if args.mode == "on" else (False,))

    for use_cache in modes:
        label = "on" if use_cache else "off"
        for index in range(args.warmup):
            benchmark.generate(args.prompt, args.max_new_tokens, use_cache)
            print(f"Warm-up cache={label} {index + 1}/{args.warmup} complete")

        measured: list[KVCacheMetrics] = []
        for repeat in range(1, args.runs + 1):
            text, metrics = benchmark.generate(
                args.prompt,
                args.max_new_tokens,
                use_cache,
            )
            measured.append(metrics)
            append_benchmark_csv(
                args.csv,
                metrics,
                experiment="02_kv_cache",
                repeat=repeat,
            )
            print(text)
            print(json.dumps(asdict(metrics), indent=2))

        if len(measured) > 1:
            ttft = [item.ttft_ms for item in measured]
            tpot = [item.tpot_ms for item in measured]
            e2e = [item.end_to_end_ms for item in measured]
            print(
                json.dumps(
                    {
                        "cache": label,
                        "summary": {
                            "runs": len(measured),
                            "ttft_ms": {
                                "p50": percentile(ttft, 50),
                                "p95": percentile(ttft, 95),
                                "p99": percentile(ttft, 99),
                            },
                            "tpot_ms": {
                                "p50": percentile(tpot, 50),
                                "p95": percentile(tpot, 95),
                                "p99": percentile(tpot, 99),
                            },
                            "end_to_end_ms": {
                                "p50": percentile(e2e, 50),
                                "p95": percentile(e2e, 95),
                                "p99": percentile(e2e, 99),
                            },
                        },
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()