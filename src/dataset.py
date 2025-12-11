from datasets import load_dataset
from .data_preprocessing import preprocess_example
from datasets import load_from_disk, DatasetDict
from pathlib import Path


def load_stackoverflow_dataset():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    ds = ds.map(lambda x: {"question": x["instruction"], "answer": x["output"]})
    cleaned = ds.map(
        preprocess_example,
        remove_columns=ds.column_names,
        num_proc=4,
    )
    cleaned = cleaned.filter(lambda x: x["valid"]).remove_columns(["valid"])  # type: ignore

    def add_hash(example):
        return {"hash": hash(example["text"])}

    cleaned = cleaned.map(add_hash)
    seen = set()
    idx = []
    for i, ex in enumerate(cleaned):
        h = ex["hash"]
        if h not in seen:
            seen.add(h)
            idx.append(i)
    cleaned = cleaned.select(idx).remove_columns(["hash"])  # type: ignore
    split = cleaned.train_test_split(test_size=0.1, seed=42)
    return cleaned, split

def load_cleaned_from_path(cleaned_root: str) -> DatasetDict:
    root = Path(cleaned_root)
    train = load_from_disk(str(root / "train"))
    test = load_from_disk(str(root / "test"))
    return DatasetDict({"train": train, "test": test})

