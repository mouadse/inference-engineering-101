"""OpenAI-compatible Qwen3.5-0.8B SGLang server on a Modal T4 GPU."""

import subprocess

import modal

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_DIR = "/models/qwen35-0.8b"


def download_model() -> None:
    """Bake the public Hugging Face weights into the Modal image."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)


image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.18-cu130")
    .entrypoint([])
    .run_function(download_model)
    .env({"SGLANG_MAMBA_CONV_DTYPE": "float16"})
)

app = modal.App("qwen35-0-8b-t4", image=image)


@app.function(
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=300,
)
@modal.web_server(8000, startup_timeout=600)
def serve() -> None:
    """Start SGLang's OpenAI-compatible server."""
    subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_DIR,
            "--served-model-name",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            "float16",
            "--context-length",
            "8192",
            "--mem-fraction-static",
            "0.8",
            "--mamba-ssm-dtype",
            "float16",
            "--num-continuous-decode-steps",
            "4",
            "--scheduler-recv-interval",
            "2",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
    )
