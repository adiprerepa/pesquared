import modal
import os
from dotenv import load_dotenv
load_dotenv()

# Build the vLLM Image
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.8.4",
        "huggingface-hub>=0.30.0",                # 🔥 updated!
        "flashinfer-python==0.2.0.post2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
).env({"VLLM_USE_V1": "1"})  # Use vLLM v1 APIs

# Model and cache setup
MODELS_DIR = "/llamas"
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Modal app definition
app = modal.App("example-vllm-openai-compatible")

# Constants
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000

# Serve function
@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("MODAL_API_KEY")],  # 🔥 inject secret
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    # Fetch API key from injected secret
    api_key = os.environ["MODAL_API_KEY"]

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", api_key,                      # 🔥 use the injected key
        "--enable-auto-tool-choice",               # 🔥 allow function calling
        "--tool-call-parser", "llama3_json",        # 🔥 tool call parser for Llama3 JSON
        # (Optional) --chat-template "/path/to/chat_template.jinja"
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
