"""OpenAI-compatible vLLM server for google/medgemma-1.5-4b-it on Modal (L4).

Usage (all via uv):
    uv sync                                  # local env with the Modal CLI
    modal secret create hf-secret HF_TOKEN=hf_...   # once; needs MedGemma license accepted
    uv run modal deploy medgemma_vllm.py      # deploy
    uv run modal run medgemma_vllm.py         # smoke-test the deployed server
    uv run bench_toks.py --url <deployed-url> # benchmark toks/sec

Model page / license: https://huggingface.co/google/medgemma-1.5-4b-it
"""

import json

import modal

MODEL_ID = "google/medgemma-1.5-4b-it"
VLLM_PORT = 8000
FAST_BOOT = False  # True = --enforce-eager (fast cold start, slower decode)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.28.0", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # fast weight downloads
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_DIR": "/root/.cache/vllm",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/vllm/torchinductor",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("medgemma-1-5-4b-vllm")


@app.server(
    image=vllm_image,
    gpu="L4",  # 24 GB: ~9 GB weights (bf16) + vision + KV cache + CUDA graphs
    unauthenticated=True,  # public URL so plain curl / bench_toks.py work
    scaledown_window=15 * 60,
    min_containers=1,  # avoid multi-minute zero-to-one GPU cold starts
    startup_timeout=20 * 60,  # cold download + torch.compile + graph capture
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],  # provides HF_TOKEN
    port=VLLM_PORT,
    target_concurrency=32,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_ID,
            "--served-model-name", MODEL_ID,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--tensor-parallel-size", "1",
            "--dtype", "auto",  # bf16 on A10G
            "--quantization", "fp8",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "8192",  # bounds KV-cache pre-allocation
            "--max-num-seqs", "32",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--limit-mm-per-prompt", json.dumps({"image": 2}),  # keep vision ON
            "--uvicorn-log-level", "info",
            "--async-scheduling",
        ]
        # Graphs + torch.compile stay on (cached on the volume) unless FAST_BOOT.
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
async def test():
    """Smoke-test: text-only + vision chat through the deployed server."""
    import aiohttp

    url = await Server.get_url.aio()
    image_url = (
        "https://cdn.britannica.com/61/93061-050-99147DCE/"
        "Statue-of-Liberty-Island-New-York-Bay.jpg"
    )
    payload = {
        "model": MODEL_ID,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.post(
            "/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
    print(body["choices"][0]["message"]["content"])
    print("usage:", body.get("usage"))
