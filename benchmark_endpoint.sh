#!/usr/bin/env bash

set -euo pipefail

endpoint_url="${ENDPOINT_URL:-https://mouadse--qwen35-0-8b-t4-serve.modal.run}"
model="${MODEL:-Qwen/Qwen3.5-0.8B}"
runs="${RUNS:-3}"
warmups="${WARMUPS:-1}"
max_tokens="${MAX_TOKENS:-256}"
prompt="${PROMPT:-Explain how a GPU executes a neural network inference workload. Be detailed and use the full available response length.}"

usage() {
    cat <<'EOF'
Usage: ./benchmark_endpoint.sh [options]

Measure client-observed generation throughput for an OpenAI-compatible chat endpoint.

Options:
  --url URL           Endpoint base URL
  --model MODEL       Model identifier sent in the request
  --runs N            Number of measured requests (default: 3)
  --warmups N         Number of unmeasured warmup requests (default: 1)
  --max-tokens N      Maximum completion tokens per request (default: 256)
  --prompt TEXT       Prompt used for every request
  -h, --help          Show this help

The same settings can be supplied with ENDPOINT_URL, MODEL, RUNS, WARMUPS,
MAX_TOKENS, and PROMPT environment variables.
EOF
}

while (($# > 0)); do
    case "$1" in
        --url)
            endpoint_url="${2:?--url requires a value}"
            shift 2
            ;;
        --model)
            model="${2:?--model requires a value}"
            shift 2
            ;;
        --runs)
            runs="${2:?--runs requires a value}"
            shift 2
            ;;
        --warmups)
            warmups="${2:?--warmups requires a value}"
            shift 2
            ;;
        --max-tokens)
            max_tokens="${2:?--max-tokens requires a value}"
            shift 2
            ;;
        --prompt)
            prompt="${2:?--prompt requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for dependency in curl jq awk; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$dependency" >&2
        exit 1
    fi
done

if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer, got: %s\n' "$runs" >&2
    exit 2
fi
if [[ ! "$warmups" =~ ^[0-9]+$ ]]; then
    printf 'WARMUPS must be a non-negative integer, got: %s\n' "$warmups" >&2
    exit 2
fi
if [[ ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_TOKENS must be a positive integer, got: %s\n' "$max_tokens" >&2
    exit 2
fi

endpoint_url="${endpoint_url%/}"
request_url="${endpoint_url}/v1/chat/completions"
benchmark_tmpdir="$(mktemp -d)"
trap 'rm -rf -- "$benchmark_tmpdir"' EXIT

payload="$benchmark_tmpdir/request.json"
jq -n \
    --arg model "$model" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{
        model: $model,
        messages: [{role: "user", content: $prompt}],
        max_tokens: $max_tokens,
        temperature: 0.7,
        top_p: 0.95
    }' >"$payload"

last_tokens=0
last_seconds=0
last_tps=0

run_request() {
    local label="$1"
    local response_file="$benchmark_tmpdir/response-${label}.json"
    local curl_metrics
    local http_status
    local elapsed_seconds
    local error_message

    curl_metrics="$(curl -sS \
        --max-time 600 \
        -o "$response_file" \
        -w '%{http_code} %{time_total}' \
        "$request_url" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload")"

    read -r http_status elapsed_seconds <<<"$curl_metrics"
    if [[ "$http_status" != "200" ]]; then
        error_message="$(jq -r '.error.message // .detail // .error // "unknown API error"' "$response_file" 2>/dev/null || true)"
        printf 'Request %s failed with HTTP %s: %s\n' "$label" "$http_status" "$error_message" >&2
        exit 1
    fi

    last_tokens="$(jq -er '.usage.completion_tokens | numbers' "$response_file")" || {
        printf 'Response %s has no numeric usage.completion_tokens field.\n' "$label" >&2
        jq . "$response_file" >&2
        exit 1
    }
    last_seconds="$elapsed_seconds"
    last_tps="$(awk -v tokens="$last_tokens" -v seconds="$last_seconds" 'BEGIN {
        if (seconds <= 0) { print "0.00" } else { printf "%.2f", tokens / seconds }
    }')"
}

printf 'Endpoint:   %s\n' "$request_url"
printf 'Model:      %s\n' "$model"
printf 'Workload:   %s measured run(s), %s warmup(s), max %s completion tokens\n\n' \
    "$runs" "$warmups" "$max_tokens"

for ((warmup_index = 1; warmup_index <= warmups; warmup_index++)); do
    printf 'Warmup %d/%d... ' "$warmup_index" "$warmups"
    run_request "warmup-${warmup_index}"
    printf '%s tokens in %.3fs (%.2f tok/s; excluded)\n' \
        "$last_tokens" "$last_seconds" "$last_tps"
done

if ((warmups > 0)); then
    printf '\n'
fi

total_tokens=0
total_seconds=0
min_tps=""
max_tps=""

printf '%-6s %10s %12s %12s\n' 'Run' 'Tokens' 'Seconds' 'Tokens/sec'
for ((run_index = 1; run_index <= runs; run_index++)); do
    run_request "run-${run_index}"
    printf '%-6d %10d %12.3f %12.2f\n' \
        "$run_index" "$last_tokens" "$last_seconds" "$last_tps"

    total_tokens=$((total_tokens + last_tokens))
    total_seconds="$(awk -v total="$total_seconds" -v current="$last_seconds" \
        'BEGIN { printf "%.6f", total + current }')"

    if [[ -z "$min_tps" ]] || awk -v current="$last_tps" -v minimum="$min_tps" \
        'BEGIN { exit !(current < minimum) }'; then
        min_tps="$last_tps"
    fi
    if [[ -z "$max_tps" ]] || awk -v current="$last_tps" -v maximum="$max_tps" \
        'BEGIN { exit !(current > maximum) }'; then
        max_tps="$last_tps"
    fi
done

aggregate_tps="$(awk -v tokens="$total_tokens" -v seconds="$total_seconds" \
    'BEGIN { printf "%.2f", tokens / seconds }')"
average_latency="$(awk -v seconds="$total_seconds" -v count="$runs" \
    'BEGIN { printf "%.3f", seconds / count }')"

printf '\nMeasured total:     %d completion tokens in %.3fs\n' "$total_tokens" "$total_seconds"
printf 'Aggregate rate:     %.2f tok/s\n' "$aggregate_tps"
printf 'Average latency:    %.3fs/request\n' "$average_latency"
printf 'Per-run rate range: %.2f-%.2f tok/s\n' "$min_tps" "$max_tps"
printf '\nNote: rates include HTTP latency but exclude warmup/cold-start requests.\n'
