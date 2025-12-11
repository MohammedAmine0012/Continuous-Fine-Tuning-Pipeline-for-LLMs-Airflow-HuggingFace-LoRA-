# Continuous Fine-Tuning Pipeline for LLMs (Airflow + Hugging Face + LoRA)

Production workflow to continuously fine-tune a base LLM with domain data, evaluate, publish to Hugging Face Hub, and deploy for inference (Triton Server; optional vLLM). Includes a minimal Chat UI.

## Features
- LoRA fine-tuning with PEFT and Transformers
- Data ingest + tokenization utilities
- Evaluation: SacreBLEU and BERTScore
- Merge LoRA adapters into full weights
- Publish merged model to Hugging Face Hub
- Triton Inference Server deployment (Python backend)
- Minimal FastAPI proxy + web UI (ChatGPT-style)
- Airflow DAG orchestrating the full pipeline

## Repository layout
```
└─ project/
   ├─ src/
   │  ├─ data_preprocessing.py   ├─ dataset.py        ├─ tokenizer.py
   │  ├─ train.py                ├─ merge.py          ├─ evaluate.py
   │  ├─ save_upload.py          ├─ inference_client.py
   │  └─ web_server.py (FastAPI -> Triton)
   ├─ model_repository/
   │  └─ deepseek_merged/
   │     ├─ config.pbtxt (backend: python, STRING I/O)
   │     └─ 1/model.py  (loads model from HF or local)
   ├─ airflow_dags/continuous_finetune.py
   ├─ webui/index.html (chat UI)
   ├─ Dockerfile.triton-hf
   ├─ requirements.txt
   └─ README.md
```

## Prerequisites
- Python 3.10+ (virtualenv recommended)
- NVIDIA GPU + CUDA drivers
- Docker
- Hugging Face account and access token

## Setup
```powershell
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt
```

Environment variables (examples):
```powershell
$env:HF_TOKEN = "hf_..."
$env:HF_MODEL_ID = "Moamineelhilali/deepseek-stackoverflow-merged"
```

## Run Triton Inference Server
```powershell
docker run --gpus=all --rm `
  --dns 1.1.1.1 --dns 8.8.8.8 `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -e HF_TOKEN=$env:HF_TOKEN `
  -e HF_MODEL_ID=$env:HF_MODEL_ID `
  -v "${PWD}\model_repository:/models" `
  my-triton-hf tritonserver --model-repository=/models
```
- Health: http://localhost:8000/v2/health/ready
- Model name: `deepseek_merged`

Quick client test:
```powershell
python src\inference_client.py --model deepseek_merged --prompt "Hello"
```

## Web Chat (FastAPI + HTML)
Run the proxy API and UI:
```powershell
uvicorn src.web_server:app --host 0.0.0.0 --port 8080
```
Open http://localhost:8080 and chat. The UI calls FastAPI, which forwards to Triton.

Environment overrides:
- `TRITON_HTTP_URL` (default `http://localhost:8000`)
- `TRITON_MODEL_NAME` (default `deepseek_merged`)

## Airflow Pipeline
The DAG `airflow_dags/continuous_finetune.py` orchestrates:
1. Ingest/tokenize data
2. LoRA fine-tuning
3. Merge adapters -> full model
4. Evaluate (BLEU, BERTScore)
5. Publish to Hugging Face
6. Deploy (Triton; optional vLLM)

Quickstart:
```powershell
$env:AIRFLOW_HOME = "$PWD\airflow_home"
airflow db init
airflow users create --username admin --password admin --firstname a --lastname b --role Admin --email admin@example.com
airflow webserver -p 8085
# in another terminal
airflow scheduler
```

Configure Variables (UI → Admin → Variables), e.g.:
- `DATA_DIR`: dataset path
- `BASE_MODEL_ID`: e.g. meta-llama/Meta-Llama-3-8B-Instruct
- `HF_TOKEN`, `HF_MERGED_REPO`: e.g. `Moamineelhilali/deepseek-stackoverflow-merged`
- `LORA_OUTPUT_DIR`, `MERGED_OUTPUT_DIR`: artifact paths
- `TRITON_MODEL_REPO`: `.../model_repository`
- `TRITON_DOCKER_IMAGE`: `my-triton-hf`

Trigger:
```powershell
airflow dags trigger continuous_finetune
```

## Hugging Face Notes
- Triton loads from `HF_MODEL_ID`; set `HF_TOKEN` for private repos.
- For offline use, mount a local snapshot and set `HF_MODEL_ID` to that path.

## Troubleshooting
- 400 invalid datatype: client must send BYTES for string tensors.
- DNS failures in Docker: add `--dns 1.1.1.1 --dns 8.8.8.8`.
- Triton backend error: ensure `backend: "python"` and correct mount path.
- Large files: model weights are excluded by `.gitignore`; pull from HF.

## License
MIT (or update per your needs).
