pip install --upgrade uv
uv venv
source .venv/bin/activate
uv pip install vllm==0.7.2 triton==3.1.0 datasets transformers==4.51.3 tensorboard torch gpustat datasets python-dotenv
uv pip install flash-attn==2.7.4.post1 --no-build-isolation