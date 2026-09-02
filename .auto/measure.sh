#!/usr/bin/env bash
set -euo pipefail

endpoint_url="${ENDPOINT_URL:-https://mouadse--qwen35-0-8b-t4-serve.modal.run}"
primary_prompt='Explain how GPU inference executes a transformer-style or hybrid language model. Discuss memory movement, kernels, and autoregressive decoding in detail. Continue until the response limit.'
holdout_prompt='You are reviewing an inference service for production. Compare continuous batching, CUDA graphs, quantization, paged KV caches, speculative decoding, and kernel fusion. For each technique, explain when it improves latency or throughput, when it can regress performance, and what correctness or quality risks must be measured. Then propose a hardware-aware benchmarking methodology that includes short and long prompts, warm and cold behavior, multiple sampling settings, concurrency levels, end-to-end HTTP latency, server-side timing, output-token throughput, time to first token, inter-token latency, memory use, and held-out workloads. Be concrete, avoid slogans, and make the analysis useful for both older Turing GPUs and newer Hopper GPUs. Continue with technical detail until the response limit.'

echo 'Deploying candidate...'
modal deploy --strategy rolling qwen35_t4_endpoint.py

primary_output="$(RUNS=3 WARMUPS=1 MAX_TOKENS=256 PROMPT="$primary_prompt" ./benchmark_endpoint.sh --url "$endpoint_url")"
printf '%s\n' "$primary_output"

holdout_output="$(RUNS=1 WARMUPS=0 MAX_TOKENS=128 PROMPT="$holdout_prompt" ./benchmark_endpoint.sh --url "$endpoint_url")"
printf '%s\n' "$holdout_output"

extract_rate() {
    awk '/^Aggregate rate:/ {print $3}' <<<"$1"
}
extract_latency_ms() {
    awk '/^Average latency:/ {gsub("s/request", "", $3); printf "%.3f", $3 * 1000}' <<<"$1"
}
extract_min_rate() {
    awk '/^Per-run rate range:/ {split($4, bounds, "-"); print bounds[1]}' <<<"$1"
}
extract_tokens() {
    awk '/^Measured total:/ {print $3}' <<<"$1"
}

printf 'METRIC output_tok_s=%s\n' "$(extract_rate "$primary_output")"
printf 'METRIC latency_ms=%s\n' "$(extract_latency_ms "$primary_output")"
printf 'METRIC min_tok_s=%s\n' "$(extract_min_rate "$primary_output")"
printf 'METRIC holdout_tok_s=%s\n' "$(extract_rate "$holdout_output")"
printf 'METRIC holdout_latency_ms=%s\n' "$(extract_latency_ms "$holdout_output")"
printf 'METRIC completion_tokens=%s\n' "$(extract_tokens "$primary_output")"
