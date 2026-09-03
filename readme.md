# MedGemma 1.5 4B on Modal with vLLM

Serves `google/medgemma-1.5-4b-it` (vision + text) as an OpenAI-compatible API
on a Modal L4 GPU, using vLLM. Includes a streaming toks/sec benchmark.

## Files

| File | Purpose |
|---|---|
| `medgemma_vllm.py` | Modal app: vLLM server + smoke test |
| `bench_toks.py` | Toks/sec benchmark (TTFT/TPOT, text + vision) |
| `pyproject.toml` | uv-managed local deps |

## Setup

```bash
uv sync
modal secret create hf-secret HF_TOKEN=hf_...   # once; accept the model license on HF first
```

## Deploy & use

```bash
uv run modal deploy medgemma_vllm.py
```

```bash
curl -X POST "https://mouadse--medgemma-1-5-4b-vllm-server.us-east.modal.direct/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{"model": "google/medgemma-1.5-4b-it",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image in one sentence."},
      {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}]}]}'
```

Local images work via base64 data URLs (`data:image/jpeg;base64,...`).

## Benchmark

```bash
uv run bench_toks.py --url https://mouadse--medgemma-1-5-4b-vllm-server.us-east.modal.direct --mode vision
uv run bench_toks.py --url https://mouadse--medgemma-1-5-4b-vllm-server.us-east.modal.direct --mode text
```

## Results (L4, bf16)

| Workload | Toks/sec (median) | TTFT | TPOT |
|---|---|---|---|
| Vision (short reply, 25 tok) | 13.7 | ~1.0 s | ~32 ms |
| Text-only (256 tok) | 29.4 | ~0.6 s | ~32 ms |

## Learnings

- **Decode is bandwidth-bound, not engine-bound.** 4B bf16 = ~8 GB weights over
  ~300 GB/s L4 bandwidth gives a ~27 ms/token floor; we measured ~32 ms.
  Switching to SGLang would not move this. The real speed lever is FP8
  quantization (`--quantization fp8`, ~2x) or a higher-bandwidth GPU.
- **Cold start is ~8 min** (weight load + torch.compile + CUDA graph capture).
  Weights and compile artifacts persist on Modal Volumes, but compile/graph
  capture rerun per fresh container. Mitigations: `FAST_BOOT=True`
  (`--enforce-eager`, faster boot, slower decode), `min_containers=1`
  (never cold, pays idle GPU), or memory snapshots.
- **L4 over A10G** for cost: 24 GB is comfortable for this model
  (~8.6 GB weights + 10 GB KV cache + graphs). T4/16 GB would be marginal
  for the vision path.
- **Keep vision on in vLLM flags**: `--limit-mm-per-prompt {"image": 2}`.
  The stock Modal example disables multimodal (`0`) — copy-pasting it would
  silently break image inputs.
- **Serving config that worked**: `--gpu-memory-utilization 0.90`,
  `--max-model-len 8192`, `--max-num-seqs 32`, prefix caching + chunked
  prefill on, graphs on (`--no-enforce-eager`).
