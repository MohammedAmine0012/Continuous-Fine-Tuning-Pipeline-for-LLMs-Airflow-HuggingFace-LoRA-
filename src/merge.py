import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .config import MODEL_ID


def merge_lora_to_base(adapter_path_or_repo: str, out_dir: str) -> str:
    """Merge a LoRA adapter into the base model and save a standalone model.
    Returns the output directory path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base, adapter_path_or_repo)
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return str(out)
