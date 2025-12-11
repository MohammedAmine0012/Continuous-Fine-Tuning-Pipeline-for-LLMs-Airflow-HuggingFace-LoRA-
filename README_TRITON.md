# Triton Serving (Merged Model from Hugging Face)

## Prerequisites
- NVIDIA GPU + drivers
- Docker
- HF token (if repo is private)

## Prepare model repository
This repo contains:
```
model_repository/
  deepseek_merged/
    config.pbtxt
    1/
      model.py
```

`model.py` pulls the merged model from Hugging Face Hub using `HF_MODEL_ID` and optional `HF_TOKEN`.

## Run Triton
```bash
docker run --gpus=all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_MODEL_ID="Moamineelhilali/deepseek-stackoverflow-merged" \
  -e TRANSFORMERS_CACHE=/models/.cache \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.11-py3 tritonserver \
  --model-repository=/models
```

## Invoke inference
```bash
python -m pip install requests
python src/inference_client.py --prompt "### Question:\n...\n\n### Answer:\n"
```

## Notes
- Prefer serving the merged model for simplest deployment.
- For private repos, `HF_TOKEN` must be provided in the environment.
- Cold start will download weights; ensure adequate disk and network.
