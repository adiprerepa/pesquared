import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "guidance>=0.1.8",  # added for structured outputs
        "jsonschema>=4.0.0",  # added for JSON schema validation
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .pip_install(
        "requests"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# Setting to VLLM_USE_V0 for better structured output compatibility with guidance
vllm_image = vllm_image.env({"VLLM_USE_V0": "1"})


MODELS_DIR = "/llamas"
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


app = modal.App("llama-3.1-8B-vllm")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY_SECRET = modal.Secret.from_name('MODAL_API_KEY')  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    secrets=[API_KEY_SECRET],
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=100
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess
    import os
    API_KEY = os.environ["MODAL_API_KEY"]
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
        "--guided-decoding-backend",
        "outlines",  # Changed from guidance -> outlines
        "--enable-lora",  # Added for potential parameter-efficient fine-tuning needs
        "--max-model-len",
        "8192",  # Setting explicit context length
        "--tensor-parallel-size",
        str(N_GPU),  # Ensuring proper GPU utilization
    ]

    subprocess.Popen(" ".join(cmd), shell=True)

