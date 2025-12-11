import os
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from .config import MODEL_ID


def load_model():
    # Ensure bitsandbytes is disabled
    os.environ["DISABLE_BNB_IMPORT"] = "1"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model

