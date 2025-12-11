import re
from bs4 import BeautifulSoup
import html2text


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text or "", "html.parser")
    plain = soup.get_text()
    h = html2text.HTML2Text()
    h.ignore_links = False
    plain = h.handle(plain)
    return (plain or "").strip()


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = clean_html(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.,:;()\[\]{}<>\=\+\-\*/#@!\?'\"_`\\|~^%&$]", "", text)
    return text.strip()


def is_valid_qa(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if len(question) < 10 or len(answer) < 20:
        return False
    if len(question) > 1000 or len(answer) > 2000:
        return False
    return True


def format_qa(question: str, answer: str) -> str:
    return f"""### Question:\n{question}\n\n### Answer:\n{answer}"""


def preprocess_example(example: dict) -> dict:
    q = clean_text(example.get("question", ""))
    a = clean_text(example.get("answer", ""))
    if is_valid_qa(q, a):
        return {"text": format_qa(q, a), "valid": True}
    return {"text": "", "valid": False}

from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import Tuple, Optional
import os

def preprocess_and_save(
    cleaned_root: str | Path,
    output_dir: str | Path,
    test_mode: bool = False,
    max_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load, preprocess, and save the dataset.
    
    Args:
        cleaned_root: Path to the cleaned dataset directory
        output_dir: Directory to save the processed dataset
        test_mode: If True, use a smaller subset for testing
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from datasets import load_from_disk
    
    # Convert to Path objects if needed
    cleaned_root = Path(cleaned_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default max_samples for test mode
    if test_mode and max_samples is None:
        max_samples = 1000

    # Load the persisted split (train/test) from cleaned_root
    from .ingest import load_cleaned_split
    split = load_cleaned_split(str(cleaned_root))

    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Limit dataset size for testing
    if test_mode and max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(max(1, max_samples // 10), len(eval_dataset))))
    
    # Save the processed datasets
    train_dataset.save_to_disk(str(output_dir / "train"))
    eval_dataset.save_to_disk(str(output_dir / "eval"))
    
    return train_dataset, eval_dataset

def preprocess():
    pass
