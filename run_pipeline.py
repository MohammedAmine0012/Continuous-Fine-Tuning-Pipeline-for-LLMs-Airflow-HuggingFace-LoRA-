import sys
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ingest import materialize_cleaned_dataset
from src.data_preprocessing import preprocess_and_save
from src.train import train
from src.evaluate import evaluate_model
from src.merge import merge_lora_to_base
from src.save_upload import push_merged_to_hub
from src.config import (
    DATA_DIR, CLEANED_DIR, ADAPTERS_DIR,
    MODEL_ID, HF_REPO_NAME, MERGED_REPO_NAME,
    timestamp_dir, QUALITY_THRESHOLDS
)

def run_pipeline(
    test_mode: bool = False,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Run the entire pipeline from data ingestion to model deployment.
    
    Args:
        test_mode: If True, use smaller dataset for testing
        push_to_hub: If True, push the final model to Hugging Face Hub
        hf_token: Hugging Face API token (required if push_to_hub is True)
        max_samples: Maximum number of samples to use in test mode
        
    Returns:
        Tuple of (success: bool, output_dir: str)
    """
    if push_to_hub and not hf_token:
        raise ValueError("HF_TOKEN is required when push_to_hub is True")
    
    # Set default max_samples if in test mode and none provided
    if test_mode and max_samples is None:
        max_samples = 1000
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ADAPTERS_DIR / f"pipeline_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/5] Starting pipeline in {output_dir}")
    
    try:
        # 1. Data Ingestion
        print(f"\n[1/5] Step 1: Ingesting and cleaning data {'(test mode)' if test_mode else ''}...")
        cleaned_root, _ = materialize_cleaned_dataset(
            test_mode=test_mode,
            max_samples=max_samples
        )
        
        # 2. Data Preprocessing
        print("\n[2/5] Step 2: Preprocessing data...")
        train_dataset, eval_dataset = preprocess_and_save(
            cleaned_root=cleaned_root,
            output_dir=output_dir / "preprocessed",
            test_mode=test_mode,
            max_samples=max_samples
        )
        
        # 3. Training
        print("\n[3/5] Step 3: Training model...")
        training_output_dir = output_dir / "training_output"
        trainer, tok, split, adapter_dir = train(
            output_dir=str(training_output_dir),
            cleaned_root=str(cleaned_root)
        )
        
        # 4. Evaluation
        print("\n[4/5] Step 4: Evaluating model...")
        eval_results = evaluate_model(
            model_path=str(adapter_dir),
            eval_dataset=eval_dataset,
            output_dir=output_dir / "evaluation"
        )
        
        # Check quality gates
        print("\n[5/5] Step 5: Checking quality gates...")
        metrics = {
            'bleu': eval_results.get('bleu', 0),
            'bertscore_f1': eval_results.get('bertscore_f1', 0)
        }
        
        print(f"\nEvaluation Results:")
        for metric, value in metrics.items():
            threshold = QUALITY_THRESHOLDS.get(metric, 0)
            status = "PASS" if value >= threshold else "FAIL"
            print(f"- {metric.upper()}: {value:.4f} (Threshold: {threshold:.2f}) [{status}]")
        # Conditional promotion vs production
        import json
        prod_dir = ADAPTERS_DIR / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        prod_metrics_path = prod_dir / "metrics.json"
        prod_info_path = prod_dir / "current.json"
        prod_adapter_ptr = prod_dir / "current_adapter.txt"

        prev = {"bleu": 0.0, "bertscore_f1": 0.0}
        if prod_metrics_path.exists():
            try:
                prev = json.loads(prod_metrics_path.read_text())
            except Exception:
                pass

        is_better = (metrics.get('bleu', 0.0) >= prev.get('bleu', 0.0)) and (metrics.get('bertscore_f1', 0.0) >= prev.get('bertscore_f1', 0.0))
        if not is_better:
            print("\nPromotion: New adapter does not beat production. Discarding (no push).")
            print("Production metrics:", prev)
            print("New metrics:", metrics)
            print(f"\n✅ Pipeline finished without promotion. Output in {output_dir}")
            return True, str(output_dir)

        print("\nPromotion: New adapter outperforms or matches production. Saving pointers and pushing adapter to Hub...")
        prod_metrics_path.write_text(json.dumps(metrics, indent=2))
        prod_adapter_ptr.write_text(str(adapter_dir))
        prod_info = {"adapter_path": str(adapter_dir), "updated_at": timestamp}
        prod_info_path.write_text(json.dumps(prod_info, indent=2))

        # Push adapter to HF Hub (requires token); try to use provided token or default login
        try:
            from huggingface_hub import HfFolder
            token = hf_token or HfFolder.get_token()
        except Exception:
            token = hf_token
        try:
            from src.save_upload import push_to_hub
            push_to_hub(trainer.model, tok, repo_name=HF_REPO_NAME, token=token)
            print(f"Pushed adapter to HF Hub repo: {HF_REPO_NAME}")
        except Exception as e:
            print(f"Warning: Failed to push adapter to HF Hub: {e}")
        
        # 5. Merge and Save
        print("\nMerging LoRA weights with base model...")
        merged_model_path = output_dir / "merged_model"
        merge_lora_to_base(adapter_path_or_repo=str(adapter_dir), out_dir=str(merged_model_path))
        
        # 6. Push to Hub (if requested)
        if push_to_hub and hf_token:
            print("\nPushing merged model to Hugging Face Hub...")
            push_merged_to_hub(
                merged_path=str(merged_model_path),
                repo_name=MERGED_REPO_NAME,
                token_env="HF_TOKEN"
            )
        
        print(f"\n✅ Pipeline completed successfully! Output in {output_dir}")
        return True, str(output_dir)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the complete model training pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to use in test mode')
    parser.add_argument('--push-to-hub', action='store_true', help='Push the final model to Hugging Face Hub')
    args = parser.parse_args()
    
    hf_token = None
    if args.push_to_hub:
        from huggingface_hub import HfFolder
        hf_token = HfFolder.get_token()
        if not hf_token:
            hf_token = input("Please enter your Hugging Face token: ")
    
    success, output_dir = run_pipeline(
        test_mode=args.test,
        push_to_hub=args.push_to_hub,
        hf_token=hf_token,
        max_samples=args.max_samples
    )
    
    if not success:
        sys.exit(1)
