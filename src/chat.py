import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from .tokenizer import load_tokenizer
from .config import MODEL_ID


def load_inference_model(adapter_path_or_repo: str | None = None):
    tok = load_tokenizer()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    if adapter_path_or_repo:
        model = PeftModel.from_pretrained(base, adapter_path_or_repo)
    else:
        model = base
    return model, tok


def ask_stackoverflow(model, tokenizer, question: str, max_tokens: int = 200) -> str:
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Answer:")[-1].strip()
    return answer

def chat(prompt: str) -> str:
    return ""
