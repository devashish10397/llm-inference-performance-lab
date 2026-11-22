"""Minimal autoregressive decoder with explicit prefill and decode phases."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    args = parser.parse_args()

    decoder = MinimalDecoder(args.model, args.device)
    text, metrics = decoder.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(text)
    print(json.dumps(asdict(metrics), indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(metrics), indent=2) + "\n")


if __name__ == "__main__":
    main()