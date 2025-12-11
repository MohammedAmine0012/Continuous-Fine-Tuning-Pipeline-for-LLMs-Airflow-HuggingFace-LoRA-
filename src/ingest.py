from pathlib import Path
from typing import Optional, Tuple
from datasets import Dataset, DatasetDict, load_from_disk
from .dataset import load_stackoverflow_dataset
from .config import CLEANED_DIR


def materialize_cleaned_dataset(
    out_dir: Optional[str] = None,
    test_mode: bool = False,
    max_samples: Optional[int] = None
) -> Tuple[str, str]:
    """Create a persisted cleaned dataset and split on disk.
    
    Args:
        out_dir: Output directory for the cleaned dataset
        test_mode: If True, use a smaller subset for testing
        max_samples: Maximum number of samples to use in test mode
        
    Returns:
        Tuple of (cleaned_root, source_description)
          - cleaned_root: root dir containing train/ and test/ subfolders
          - source: a short description of source used
    """
    # Set default max_samples if in test mode and none provided
    if test_mode and max_samples is None:
        max_samples = 1000
        
    cleaned_root = Path(out_dir) if out_dir else (CLEANED_DIR / "run")
    cleaned_root.mkdir(parents=True, exist_ok=True)

    # Load and preprocess the dataset
    cleaned, split = load_stackoverflow_dataset()
    
    # Limit dataset size for testing
    if test_mode and max_samples is not None:
        train_size = min(max_samples, len(split["train"]))
        test_size = min(max(1, max_samples // 10), len(split["test"]))
        split["train"] = split["train"].select(range(train_size))
        split["test"] = split["test"].select(range(test_size))

    # Persist both splits to disk
    train_dir = cleaned_root / "train"
    test_dir = cleaned_root / "test"
    split["train"].save_to_disk(str(train_dir))
    split["test"].save_to_disk(str(test_dir))

    return str(cleaned_root), f"alpaca_18k_remote{' (test mode)' if test_mode else ''}"


def load_cleaned_split(cleaned_root: str) -> DatasetDict:
    root = Path(cleaned_root)
    train = load_from_disk(str(root / "train"))
    test = load_from_disk(str(root / "test"))
    return DatasetDict({"train": train, "test": test})
