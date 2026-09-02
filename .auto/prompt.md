# Autoresearch: maximize Qwen3.5-0.8B inference throughput on Modal T4

## Objective
Maximize client-observed output tokens/second for the existing OpenAI-compatible `Qwen/Qwen3.5-0.8B` endpoint on one NVIDIA T4. Investigate TensorRT-LLM first, using the NVIDIA TensorRT-LLM performance skills and upstream implementation/docs, but retain another engine when TensorRT-LLM does not support this exact dense hybrid model or Turing GPU correctly.

## Metrics
- **Primary**: `output_tok_s` (tokens/s, higher is better) — aggregate completion tokens divided by end-to-end HTTP request time over three warmed requests generating up to 256 tokens.
- **Secondary**: `latency_ms`, `min_tok_s`, `holdout_tok_s`, `holdout_latency_ms`, and `completion_tokens`. The holdout uses a materially longer, different prompt and a 128-token generation.

## How to Run
`./.auto/measure.sh` deploys the current Modal app, warms it, benchmarks the fixed primary workload, then benchmarks the holdout. It emits `METRIC name=value` lines.

## Files in Scope
- `qwen35_t4_endpoint.py` — Modal image, GPU allocation, and inference server configuration. This is the optimization target.
- `.auto/prompt.md` — update the experiment history when useful.
- `.auto/ideas.md` — deferred optimization backlog.

## Off Limits
- `benchmark_endpoint.sh` — frozen benchmark client.
- `.auto/measure.sh` — frozen workload and metric extraction after baseline.
- `.auto/checks.sh` — frozen correctness backpressure after baseline.
- Model identity, one-T4 GPU allocation, prompts, token limits, request count, completion-token accounting, and endpoint response content.

## Constraints
- Keep `Qwen/Qwen3.5-0.8B`, one `T4`, FP16/BF16-or-better output quality, and an OpenAI-compatible `/v1/chat/completions` endpoint.
- Do not reduce generated tokens, truncate responses, cache/replay benchmark answers, detect benchmark prompts, alter metric accounting, skip work, or specialize code to benchmark text.
- Do not overfit: monitor the independent long-prompt holdout and prefer engine-wide improvements. A primary win with a catastrophic holdout regression is not acceptable.
- Every response must be non-empty, report numeric token usage, and pass the existing benchmark client's HTTP/schema checks.
- Use official framework images directly where practical; do not overlay conflicting SGLang/PyTorch installations.

## What's Been Tried
- Initial setup is SGLang 0.5.18, FP16, 8192 context, 0.8 static memory fraction, Mamba FP16, and prefill CUDA graphs disabled.
- Upstream TensorRT-LLM main contains a Qwen3.5 implementation, but its model-coverage registry currently disables dense `Qwen/Qwen3.5-0.8B` due an open `use_cache` configuration failure (NVIDIA/TensorRT-LLM#14672). Validate rather than assuming it works.
