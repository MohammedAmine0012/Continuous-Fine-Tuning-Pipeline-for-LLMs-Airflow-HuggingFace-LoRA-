# 🚀 StackOverflow Assistant: Continuous Fine-Tuning Pipeline

A production-grade **Self-Improving AI System** that continuously fine-tunes a Large Language Model (DeepSeek Coder) on new domain-specific data, evaluates its quality, and automatically deploys it for high-performance inference.

![Status](https://img.shields.io/badge/Status-Operational-success)
![Stack](https://img.shields.io/badge/Stack-Airflow%20%7C%20Triton%20%7C%20LoRA-blue)

---

## 🏗️ Architecture

The system is designed as a closed-loop pipeline:

```mermaid
graph LR
    A[New Data] -->|1. Ingest| B[Airflow Pipeline]
    B -->|2. Fine-Tune (LoRA)| C[DeepSeek Coder]
    C -->|3. Evaluate (BLEU/BERT)| D{Quality Gate}
    D -- Pass --> E[Merge & Deploy]
    D -- Fail --> F[Discard]
    E -->|4. Push| G[Hugging Face Hub]
    G -->|5. Pull| H[Triton Inference Server]
    H -->|6. Serve| I[Web Interface]
```

### Components
1.  **Airflow Orchestrator**: Manages the end-to-end workflow (Ingest → Train → Merge → Deploy).
2.  **Fine-Tuning Engine**: Uses **LoRA** (Low-Rank Adaptation) for efficient, low-cost training on new data.
3.  **Deployment Target**: **Hugging Face Hub** acts as the Model Registry / Source of Truth.
4.  **Inference Server**: **NVIDIA Triton Inference Server** (running in Docker) provides high-performance serving.
5.  **Web Interface**: A modern Glassmorphism-style chat UI (FastAPI + HTML/JS) for user interaction.

---

## 🚀 Getting Started

### Prerequisites
*   **NVIDIA GPU**: Minimum 24GB VRAM recommended (A10G, A100, or RTX 3090/4090).
*   **Docker & Docker Compose**: For running the infrastructure.
*   **Python 3.10+**: For local development.
*   **Hugging Face Object**: A Read/Write access token.

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/stackoverflow-assistant.git
cd stackoverflow-assistant

# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Python deps
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_token_here
HF_REPO_NAME=your_username/deepseek-stackoverflow-merged
TRITON_HTTP_URL=http://localhost:8000
```

---

## 🏃 Usage

### A. Running the Chat Interface (Manual Mode)
If you just want to chat with the AI:
```bash
# 1. Start the Web Server
python -m uvicorn src.web_server:app --port 8080 --reload
```
Open **http://localhost:8080** in your browser.

### B. Running the Inference Server (Triton)
To serve the model with high performance:
```powershell
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### C. Running the Automation Pipeline (Airflow)
To start the continuous fine-tuning loop:
```bash
docker-compose up -d
```
Access the Airflow UI at **http://localhost:8085**.
1.  Enable the `continuous_finetune` DAG.
2.  Trigger it manually or set a schedule.

---

## 📂 Project Structure

| Path | Description |
| :--- | :--- |
| `airflow_dags/` | Airflow pipeline definitions (DAGs). |
| `src/` | Core source code (Training, Ingestion, API). |
| `src/train.py` | LoRA Fine-Tuning logic. |
| `src/merge.py` | Merges LoRA adapters into the base model. |
| `src/web_server.py` | FastAPI backend for the Chat UI. |
| `webui/` | Frontend Assets (HTML/CSS/JS). |
| `model_repository/` | Triton Server configuration. |

---

## 🛠️ Deployment on Azure

This project is Cloud-Native ready.

*   **Frontend**: Deploy `src/web_server.py` to **Azure App Service**.
*   **Backend**: Deploy the Triton Docker container to **Azure Kubernetes Service (AKS)** or **Azure Container Instances (ACI)**.
*   **Storage**: Use Azure Files for the model repository or pull directly from Hugging Face.

---

## 📄 License
MIT License. Free to use and modify.
