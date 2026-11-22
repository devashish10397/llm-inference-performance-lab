"""Minimal autoregressive decoder with explicit prefill and decode phases."""

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


@dataclass
class GenerationMetrics:
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
    max_new_tokens: int = 32
    gpu_name: str | None = None
    torch_version: str = torch.__version__


class MinimalDecoder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> tuple[str, GenerationMetrics]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_tokens = input_ids.shape[1]

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - start) * 1000

            past_key_values = outputs.past_key_values
            first_token_start = time.perf_counter()
            next_token = self._next_token(outputs.logits[:, -1, :], temperature)
            if self.device == "cuda":
                torch.cuda.synchronize()
            first_token_ms = (time.perf_counter() - first_token_start) * 1000
            ttft_ms = prefill_ms + first_token_ms
            generated_ids = [next_token]

            decode_times: list[float] = []
            for _ in range(max_new_tokens - 1):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                step_start = time.perf_counter()
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
                if self.device == "cuda":
                    torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - step_start) * 1000)
                past_key_values = outputs.past_key_values
                next_token = self._next_token(outputs.logits[:, -1, :], temperature)
                generated_ids.append(next_token)

        end_to_end_ms = (time.perf_counter() - start) * 1000
        generated = torch.cat(generated_ids, dim=-1).to("cpu")
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        tpot_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if self.device == "cuda"
            else None
        )

        metrics = GenerationMetrics(
            model=self.model_name,
            device=self.device,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            prefill_ms=prefill_ms,
            ttft_ms=ttft_ms,
            end_to_end_ms=end_to_end_ms,
            tpot_ms=tpot_ms,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            max_new_tokens=max_new_tokens,
            gpu_name=(
                torch.cuda.get_device_name(0)
                if self.device == "cuda"
                else None
            ),
        )
        return text, metrics

    def _next_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal LLM generation benchmark")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--prompt", default="Large language model inference")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/01_baseline/runs.csv"),
        help="Append one run to this CSV",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--experiment", default="01_baseline")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--prompt-token-counts",
        type=str,
        default=None,
        help="Comma-separated prompt token targets, for example: 4,128,512",
    )
    parser.add_argument(
        "--output-token-counts",
        type=str,
        default=None,
        help="Comma-separated output token targets, for example: 8,32,128",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    prompt_counts = (
        [int(value.strip()) for value in args.prompt_token_counts.split(",")]
        if args.prompt_token_counts
        else [None]
    )
    output_counts = (
        [int(value.strip()) for value in args.output_token_counts.split(",")]
        if args.output_token_counts
        else [args.max_new_tokens]
    )
    if any(count < 1 for count in prompt_counts if count is not None):
        parser.error("prompt token counts must be at least 1")
    if any(count < 1 for count in output_counts):
        parser.error("output token counts must be at least 1")

    decoder = MinimalDecoder(args.model, args.device)
    for prompt_count in prompt_counts:
        sweep_prompt = args.prompt
        if prompt_count is not None:
            while len(decoder.tokenizer(sweep_prompt)["input_ids"]) < prompt_count:
                sweep_prompt += " large language model inference"

        for output_count in output_counts:
            for warmup_index in range(args.warmup):
                decoder.generate(
                    sweep_prompt,
                    max_new_tokens=output_count,
                    temperature=args.temperature,
                )
                print(f"Warm-up {warmup_index + 1}/{args.warmup} complete")

            measured: list[GenerationMetrics] = []
            for run_index in range(args.runs):
                text, metrics = decoder.generate(
                    sweep_prompt,
                    max_new_tokens=output_count,
                    temperature=args.temperature,
                )
                repeat_number = args.repeat + run_index
                measured.append(metrics)
                print(text)
                print(json.dumps(asdict(metrics), indent=2))
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(asdict(metrics), indent=2) + "\n"
                    )
                if args.csv:
                    append_benchmark_csv(
                        args.csv,
                        metrics,
                        experiment=args.experiment,
                        repeat=repeat_number,
                    )

            if len(measured) > 1:
                ttft = [metric.ttft_ms for metric in measured]
                tpot = [metric.tpot_ms for metric in measured]
                e2e = [metric.end_to_end_ms for metric in measured]
                print(
                    json.dumps(
                        {
                            "prompt_tokens_target": prompt_count,
                            "output_tokens_target": output_count,
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