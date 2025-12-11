from transformers import TrainingArguments
from trl import SFTTrainer
from .dataset import load_stackoverflow_dataset, load_cleaned_from_path
from .tokenizer import load_tokenizer
from .model import load_model
from .config import TRAIN_PARAMS, BASE_DIR
from .save_upload import save_adapter


def build_training_args(output_dir: str) -> TrainingArguments:
    # Normalize strategies vs steps for transformers 4.57.x
    eval_steps_cfg = TRAIN_PARAMS.get("eval_steps", None)
    save_steps_cfg = TRAIN_PARAMS.get("save_steps", None)

    # Only set steps when > 0; do not pass explicit strategies to avoid version-specific conflicts
    eval_steps = eval_steps_cfg if (isinstance(eval_steps_cfg, int) and eval_steps_cfg > 0) else None
    # transformers 4.57.3 defaults save_strategy to 'steps' and expects save_steps to be an int.
    # If user disabled saving (0 or None), force a minimal integer (1) to avoid None > 1 comparisons in __post_init__.
    save_steps = save_steps_cfg if (isinstance(save_steps_cfg, int) and save_steps_cfg > 0) else 1

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_PARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_PARAMS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_PARAMS["gradient_accumulation_steps"],
        learning_rate=TRAIN_PARAMS["learning_rate"],
        num_train_epochs=TRAIN_PARAMS["num_train_epochs"],
        fp16=TRAIN_PARAMS["fp16"],
        logging_steps=TRAIN_PARAMS["logging_steps"],
        eval_steps=eval_steps,
        save_steps=save_steps,
        # rely on defaults for evaluation/save strategies
        load_best_model_at_end=TRAIN_PARAMS["load_best_model_at_end"],
        metric_for_best_model=TRAIN_PARAMS["metric_for_best_model"],
        warmup_ratio=TRAIN_PARAMS["warmup_ratio"],
        lr_scheduler_type=TRAIN_PARAMS["lr_scheduler_type"],
        gradient_checkpointing=TRAIN_PARAMS["gradient_checkpointing"],
        max_grad_norm=TRAIN_PARAMS["max_grad_norm"],
        report_to=TRAIN_PARAMS["report_to"],
        save_total_limit=TRAIN_PARAMS["save_total_limit"],
    )


def train(output_dir: str | None = None, cleaned_root: str | None = None):
    if cleaned_root:
        split = load_cleaned_from_path(cleaned_root)
    else:
        _, split = load_stackoverflow_dataset()
    tok = load_tokenizer()
    model = load_model()

    odir = output_dir or str(BASE_DIR / "adapters" / "training_output")
    args = build_training_args(odir)

    # TRL API compatibility: new versions expect processing_class, older expect tokenizer/dataset_text_field
    try:
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tok,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tok,
            dataset_text_field="text",
            max_seq_length=512,
        )

    trainer.train()
    # Persist LoRA adapter for downstream tasks
    adapter_dir = save_adapter(trainer.model, tok)
    return trainer, tok, split, adapter_dir

