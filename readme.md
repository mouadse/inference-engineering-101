# MedGemma 27B on Modal with vLLM

Serves `google/medgemma-27b-it` (vision + text) as an OpenAI-compatible API
on one Modal A100 GPU. The deployment uses FP8 weight quantization so the 27B
checkpoint fits an A100 with room for vision inference and the KV cache.

## Files

| File | Purpose |
|---|---|
| `medgemma_vllm.py` | Modal deployment: FP8 vLLM server and smoke test |
| `benchmark_medgemma.py` | Concurrent production benchmark: P50/P90/P99 TTFT, TPOT, total latency, throughput, and quality checks |
| `bench_toks.py` | Quick serial streaming benchmark for text or vision |
| `test_medgemma_vllm.py` | Unit test for the server startup-wait helper |
| `web/` | Next.js clinical chat UI proxying the Modal endpoint |
| `xray.jpg` | Local vision input used by the production benchmark |
| `pyproject.toml` | uv-managed local dependencies |
| `.env.example` / `web/.env.example` | Required env vars (`HF_TOKEN`, `MEDGEMMA_API_URL`) |

## Current production configuration

| Setting | Value | Why |
|---|---|---|
| Model | `google/medgemma-27b-it` | 27B instruction-tuned text-and-vision checkpoint |
| GPU | One NVIDIA A100 (40 GB minimum) | Modal may transparently upgrade `gpu="A100"` to an 80 GB A100 |
| Engine | vLLM `0.28.0` | Continuous batching, PagedAttention, prefix caching, and OpenAI-compatible streaming |
| Precision | FP8 weights with BF16 activations (`--quantization fp8`) | Ampere uses vLLM's Marlin W8A16 path, reducing model weight memory by about half |
| Context limit | 8,192 tokens | Bounds KV-cache allocation |
| Request capacity | 32 sequences / target concurrency 32 | Caps scheduler concurrency and KV-cache demand |
| Compilation | CUDA graphs and `torch.compile` enabled | Better steady-state decode performance |
| Availability | `min_containers=1` | Keeps one A100 warm and prevents normal requests from paying the GPU cold start |

The always-warm container bills continuously. Set `min_containers=0` only when
idle cost matters more than request latency.

## Setup

```bash
uv sync
set -a && . ./.env && set +a && uv run modal secret create --force hf-secret HF_TOKEN="$HF_TOKEN"
```

## Deploy & use

```bash
uv run modal deploy medgemma_vllm.py
```

```bash
curl -X POST "https://mouadse--medgemma-27b-vllm-server.us-east.modal.direct/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{"model": "google/medgemma-27b-it",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image in one sentence."},
      {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}]}]}'
```

Local images work via base64 data URLs (`data:image/jpeg;base64,...`).

## Benchmark

Use the small serial benchmark for a quick text or vision check:

```bash
uv run bench_toks.py --url https://mouadse--medgemma-27b-vllm-server.us-east.modal.direct --mode vision
uv run bench_toks.py --url https://mouadse--medgemma-27b-vllm-server.us-east.modal.direct --mode text
```

Use the concurrent benchmark for production latency percentiles:

```bash
uv run python benchmark_medgemma.py \
  --url https://mouadse--medgemma-27b-vllm-server.us-east.modal.direct \
  --modes text,vision \
  --concurrency 1,4,16 \
  --requests 100 \
  --max-tokens 64 \
  --output benchmark-results.json
```

The full benchmark sends 600 measured requests plus warmups. It deliberately
takes several minutes so each workload has enough samples for a meaningful P99.
For a fast smoke check, use `--concurrency 1 --requests 1`; do not treat that
single sample as a latency percentile.

## Web clinical UI

`web/` is a Next.js chat UI for text questions and medical-image review. It
posts to `/api/chat`, which injects a short system prompt (clinical for text,
radiology-structured for images), then proxies the Modal endpoint with
streaming at temperature 0. Images must be JPG, PNG, or WebP under 3 MB and
are sent as base64 data URLs.

```bash
cd web
npm install
cp .env.example .env  # set MEDGEMMA_API_URL, or rely on the deployed default
npm run dev            # npm run build && npm start for production
```

## Tests

```bash
.venv/bin/python test_medgemma_vllm.py  # server startup-wait helper
node --test web/tests/chat-route.test.mjs  # chat route prompts (run from repo root)
npm run typecheck --prefix web
```

## Historical 4B/L4 optimization result

The measurements below are retained from the former
`google/medgemma-1.5-4b-it` deployment on an L4; they do not characterize the
current 27B/A100 deployment. The primary score was the geometric mean of warm,
client-observed P99 TTFT across text and vision at concurrency 1, 4, and 16.
Lower is better.

| Primary metric | BF16 baseline | Optimized FP8 | Improvement |
|---|---:|---:|---:|
| P99 TTFT geometric mean | 1,263.9 ms | 935.1 ms | **26.0% faster** |

Each row below contains 100 measured streaming requests after warmup, with a
64-token output limit and persistent client connections.

### Tail latency

| Workload | P99 TTFT, BF16 → FP8 | Change | P99 total, BF16 → FP8 | Change |
|---|---:|---:|---:|---:|
| Text, concurrency 1 | 975.8 → 498.1 ms | **49.0% faster** | 2,916.5 → 2,482.0 ms | **14.9% faster** |
| Text, concurrency 4 | 1,784.4 → 501.0 ms | **71.9% faster** | 3,887.3 → 2,534.4 ms | **34.8% faster** |
| Text, concurrency 16 | 539.5 → 1,795.4 ms | 232.8% slower | 2,666.0 → 3,193.5 ms | 19.8% slower |
| Vision, concurrency 1 | 754.4 → 826.1 ms | 9.5% slower | 2,758.8 → 2,165.6 ms | **21.5% faster** |
| Vision, concurrency 4 | 2,311.1 → 2,040.7 ms | **11.7% faster** | 4,387.9 → 3,414.5 ms | **22.2% faster** |
| Vision, concurrency 16 | 2,488.3 → 884.9 ms | **64.4% faster** | 4,737.1 → 3,659.9 ms | **22.7% faster** |

### Throughput

| Workload | BF16 tok/s | FP8 tok/s | Improvement |
|---|---:|---:|---:|
| Text, concurrency 1 | 26.5 | 28.8 | **+8.6%** |
| Text, concurrency 4 | 98.8 | 109.4 | **+10.7%** |
| Text, concurrency 16 | 382.8 | 478.6 | **+25.0%** |
| Vision, concurrency 1 | 25.4 | 33.7 | **+32.5%** |
| Vision, concurrency 4 | 95.0 | 125.2 | **+31.7%** |
| Vision, concurrency 16 | 300.5 | 391.2 | **+30.2%** |

FP8 increased throughput in every tested workload by **8.6–32.5%**. Tail
latency did not improve uniformly: text at concurrency 16 and vision at
concurrency 1 regressed, so production monitoring should preserve the same
per-workload breakdown instead of watching only the aggregate score.

## Historical 4B/L4 experiment log

| Experiment | Primary metric | Decision | What happened |
|---|---:|---|---|
| vLLM BF16 baseline | 1,263.856 ms | Baseline | CUDA graphs, prefix caching, and chunked prefill already enabled |
| Online FP8 W8A8 | **935.064 ms** | **Kept** | Primary P99 TTFT improved 26.0%; every throughput cell improved |
| Reserve six vCPUs | 979.808 ms | Rejected | Throughput improved, but the primary P99 metric regressed 4.8% versus FP8 |
| Raise prefill budget to 8,192 | 915.215 ms | Rejected | Aggregate score improved 2.1%, but vision concurrency-16 P99 reached 6.5 s and throughput fell 30.0% |
| Disable Uvicorn access logs | 1,072.938 ms | Rejected | Primary P99 metric regressed 14.7% |
| FP8 E4M3 KV cache | 1,000.765 ms | Rejected | Quality and throughput passed, but primary P99 metric regressed 7.0% |
| TensorRT-LLM 1.3.0rc25 | Invalid | Rejected | Gemma 3 attention warmup crashed before the server became ready |

The TensorRT-LLM trial failed with:

```text
AttributeError: 'TrtllmAttentionMetadata' object has no attribute
'swap_paged_kv_indices_for_layer'
```

A rolling deployment initially made this candidate look successful because
Modal continued routing benchmark traffic to the old vLLM container. The
zero-to-one test exposed the crash, and the TensorRT result was invalidated.

## Practical performance guidance
Only FP8 is capacity-driven for the new deployment. The remaining tuning is
inherited from the former 4B/L4 configuration and must be rebenchmarked on the
27B/A100 pair.

| Do | Why |
|---|---|
| Keep `--quantization fp8` | The 27B checkpoint needs reduced weight memory on a single 40 GB A100; vLLM uses weight-only W8A16 on Ampere |
| Keep CUDA graphs enabled | `--no-enforce-eager` pays more startup cost but improves steady-state decode |
| Keep prefix caching enabled | Reuses KV state for shared prompt prefixes instead of repeating prefill work |
| Keep chunked prefill enabled at its default budget | Raising the budget to 8,192 caused a severe vision tail-latency regression |
| Keep the multimodal limit at `{\"image\": 2}` | Setting it to zero silently disables image requests |
| Reuse HTTP/TLS connections | The production benchmark uses one persistent `aiohttp` session; clients should do the same |
| Stream completions | Users see the first token instead of waiting for the entire medical response |
| Keep one container warm for latency-sensitive production | Avoids exposing users to model download, load, and compilation time after scale-to-zero |
| Track text and vision separately at each concurrency | The aggregate score hid regressions in two individual P99 TTFT cells |

## Historical 4B/L4 rejected changes

These findings apply to the former 4B/L4 benchmark, not the current
27B/A100 deployment.

- **Do not use TensorRT-LLM 1.3.0rc25 for that checkpoint.** Its advertised
  Gemma 3 architecture support did not survive MedGemma's attention warmup on
  the L4.
- **Do not assume more CPU lowers latency.** Six vCPUs raised throughput but
  worsened the chosen P99 latency metric.
- **Do not enable FP8 KV cache just because FP8 weights helped.** These short
  prompts were not KV-memory-bound; conversion overhead made P99 worse.
- **Do not optimize only the aggregate average.** The 8,192-token prefill
  candidate looked slightly better in aggregate while making vision P99
  unacceptable.
- **Do not use `FAST_BOOT=True` in the always-warm production deployment.**
  Eager mode boots faster but gives up CUDA-graph steady-state performance.

## Quality and safety

The former 4B/L4 FP8 candidate passed deterministic text and chest-X-ray smoke
checks and matched the recorded BF16 answers. Those checks detect broken text,
vision, and API paths; they are not a clinical-accuracy evaluation. Run the
same checks plus a representative medical validation set for the 27B/A100
deployment before using its quantized output in a clinical workflow.

## References

- [vLLM FP8 quantization](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)
- [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Modal Server autoscaling and cold starts](https://modal.com/docs/guide/servers)
- [TensorRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html)
