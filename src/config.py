from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
CLEANED_DIR = DATA_DIR / 'cleaned'
ADAPTERS_DIR = BASE_DIR / 'adapters'

# DeepSeek model used in the notebook
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
TOKENIZER_ID = MODEL_ID

# Default repo to push (customize as needed)
HF_REPO_NAME = "Moamineelhilali/deepseek-stackoverflow-adapter"
MERGED_REPO_NAME = "Moamineelhilali/deepseek-stackoverflow-merged"

def timestamp_dir(prefix: str, base: Path = ADAPTERS_DIR) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{prefix}_{ts}"

# Default training parameters (FP16, no bitsandbytes)
TRAIN_PARAMS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "fp16": True,
    "logging_steps": 10,
    "eval_steps": 0,
    "save_steps": 0,
    "eval_strategy": "no",
    "save_strategy": "no",
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "save_total_limit": 1,
}

# Evaluation thresholds for promotion
QUALITY_THRESHOLDS = {
    "bleu": 0.10,
    "bertscore_f1": 0.80,
}
