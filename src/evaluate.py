from typing import List, Tuple, Dict
import evaluate as hf_evaluate
from bert_score import score as bert_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .config import MODEL_ID, TOKENIZER_ID


def generate_predictions(model, tokenizer, test_dataset, limit: int = 100) -> Tuple[List[str], List[List[str]]]:
    preds: List[str] = []
    refs: List[List[str]] = []
    subset = test_dataset.select(range(min(limit, len(test_dataset))))
    model.eval()
    for sample in tqdm(subset, desc="Evaluating"):
        text = sample["text"]
        question = text.split("### Answer:")[0].replace("### Question:", "").strip()
        reference = text.split("### Answer:")[-1].strip()
        prompt = f"### Question:\n{question}\n\n### Answer:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_output.split("### Answer:")[-1].strip()
        preds.append(prediction)
        refs.append([reference])
    return preds, refs


def compute_bleu(preds: List[str], refs: List[List[str]]) -> Dict:
    bleu = hf_evaluate.load("bleu")
    return bleu.compute(predictions=preds, references=refs)


def compute_bertscore(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    P, R, F1 = bert_score(preds, [r[0] for r in refs], lang="en", verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


def plot_metrics(bleu_score: Dict, bert_results: Dict[str, float]):
    bleu_value = bleu_score.get("bleu", 0.0)
    precisions = bleu_score.get("precisions", [0, 0, 0, 0])
    metrics = ["Precision", "Recall", "F1"]
    values = [bert_results.get("precision", 0), bert_results.get("recall", 0), bert_results.get("f1", 0)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    bars = plt.bar([f"BLEU-{i+1}" for i in range(4)], precisions, color="skyblue")
    plt.axhline(y=bleu_value, color="red", linestyle="--", label=f"Overall BLEU = {bleu_value:.4f}")
    plt.title("BLEU Score Breakdown")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    for bar, val in zip(bars, precisions):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}", ha="center")

    plt.subplot(1, 2, 2)
    bars = plt.bar(metrics, values, color="lightgreen")
    plt.title("BERTScore Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}", ha="center")
    plt.tight_layout()
    plt.show()

def evaluate():
    pass


def evaluate_model(model_path: str, eval_dataset, output_dir) -> Dict[str, float]:
    """
    Load the base model + LoRA adapter from model_path, generate predictions on eval_dataset,
    and compute BLEU and BERTScore F1. Returns a metrics dict.
    """
    # Load tokenizer and model with adapter
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, model_path)

    # Speed: eval a small subset by default inside generate_predictions (limit)
    preds, refs = generate_predictions(model, tok, eval_dataset, limit=50)

    # Compute metrics
    bleu = compute_bleu(preds, refs)
    bert = compute_bertscore(preds, refs)
    metrics = {
        "bleu": float(bleu.get("bleu", 0.0)),
        "bertscore_precision": float(bert.get("precision", 0.0)),
        "bertscore_recall": float(bert.get("recall", 0.0)),
        "bertscore_f1": float(bert.get("f1", 0.0)),
    }
    return metrics
