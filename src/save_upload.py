import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from huggingface_hub import login
from .config import ADAPTERS_DIR, timestamp_dir, HF_REPO_NAME


def save_adapter(model, tokenizer, out_dir: Optional[str] = None) -> Path:
    out = Path(out_dir) if out_dir else timestamp_dir("adapter", ADAPTERS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def zip_dir(path: Path) -> Path:
    zip_path = path.with_suffix("").as_posix() + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.make_archive(zip_path, 'zip', str(path))
    return Path(zip_path + ".zip")


def push_to_hub(model, tokenizer, repo_name: Optional[str] = None, token: Optional[str] = None):
    if token:
        login(token=token)
    repo = repo_name or HF_REPO_NAME
    model.push_to_hub(repo, commit_message="Upload LoRA adapter")
    tokenizer.push_to_hub(repo)


def push_merged_to_hub(merged_path: str, repo_name: str, token_env: str = "HF_TOKEN"):
    """Push a merged model folder to HF Hub. Reads token from env var by default."""
    token = os.environ.get(token_env)
    if token:
        login(token=token)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(merged_path)
    tok = AutoTokenizer.from_pretrained(merged_path)
    model.push_to_hub(repo_name, commit_message="Upload merged model")
    tok.push_to_hub(repo_name)

def save_and_upload():
    pass
