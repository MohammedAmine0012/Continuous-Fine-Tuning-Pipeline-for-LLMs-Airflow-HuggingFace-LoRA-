# Central defaults for the Airflow pipeline. Edit this single file to change defaults.
# Airflow Variables (if set) will override these at runtime, but these are the fallbacks.

from src.config import MODEL_ID as CONFIG_MODEL_ID, MERGED_REPO_NAME as CONFIG_MERGED_REPO

# Paths inside the Airflow containers (match docker-compose mounts)
DATA_DIR = "/opt/airflow/data"
LORA_OUTPUT_DIR = "/opt/airflow/adapters/{{ ds_nodash }}"
MERGED_OUTPUT_DIR = "/opt/airflow/adapters/{{ ds_nodash }}/merged_model"
TRITON_MODEL_REPO = "/opt/airflow/model_repository"
TRITON_DOCKER_IMAGE = "my-triton-hf"
TRITON_HTTP_URL = "http://localhost:8000"
TRITON_MODEL_NAME = "deepseek_merged"

# Models / repos
BASE_MODEL_ID = CONFIG_MODEL_ID  # default base model (can be overridden by Airflow Variable BASE_MODEL_ID)
HF_MERGED_REPO = CONFIG_MERGED_REPO  # default merged repo (can be overridden by Airflow Variable HF_MERGED_REPO)

# Evaluation gate thresholds
BLEU_MIN = 15.0
BERTSCORE_F1_MIN = 0.78
