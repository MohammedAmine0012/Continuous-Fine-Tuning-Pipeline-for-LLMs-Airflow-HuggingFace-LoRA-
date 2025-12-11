# -*- coding: utf-8 -*-
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Project modules
from src.ingest import materialize_cleaned_dataset
from src.train import train as train_pipeline
from src.evaluate import generate_predictions, compute_bleu, compute_bertscore
from src.merge import merge_lora_to_base
from src.save_upload import push_merged_to_hub
from src.config import QUALITY_THRESHOLDS, ADAPTERS_DIR, MERGED_REPO_NAME


def run_ingest(**context):
    cleaned_root, source = materialize_cleaned_dataset()
    context['ti'].xcom_push(key='cleaned_root', value=cleaned_root)
    context['ti'].xcom_push(key='data_source', value=source)


def run_training(**context):
    ti = context['ti']
    cleaned_root = ti.xcom_pull(key='cleaned_root', task_ids='ingest_preprocess')
    trainer, tok, split, adapter_dir = train_pipeline(cleaned_root=cleaned_root)
    ti.xcom_push(key='adapter_dir', value=str(adapter_dir))
    # Optionally push counts
    ti.xcom_push(key='split_sizes', value={
        'train': len(split['train']),
        'test': len(split['test'])
    })


def run_evaluation(**context):
    ti = context['ti']
    cleaned_root = ti.xcom_pull(key='cleaned_root', task_ids='ingest_preprocess')
    adapter_dir = ti.xcom_pull(key='adapter_dir', task_ids='train_model')

    # Reload minimal state for evaluation
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from src.config import MODEL_ID
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Load dataset from disk
    from src.ingest import load_cleaned_split
    split = load_cleaned_split(cleaned_root)

    preds, refs = generate_predictions(model, tok, split['test'], limit=100)
    bleu = compute_bleu(preds, refs)
    bert = compute_bertscore(preds, refs)
    ti.xcom_push(key='bleu', value=bleu)
    ti.xcom_push(key='bertscore', value=bert)


def quality_gate(**context):
    ti = context['ti']
    bleu = ti.xcom_pull(key='bleu', task_ids='evaluate_model') or {}
    bert = ti.xcom_pull(key='bertscore', task_ids='evaluate_model') or {}
    bleu_ok = bleu.get('bleu', 0) >= QUALITY_THRESHOLDS['bleu']
    bert_ok = bert.get('f1', 0) >= QUALITY_THRESHOLDS['bertscore_f1']
    return 'merge_lora' if (bleu_ok and bert_ok) else 'stop_pipeline'


def run_merge(**context):
    ti = context['ti']
    adapter_dir = ti.xcom_pull(key='adapter_dir', task_ids='train_model')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = str(ADAPTERS_DIR / f'merged_{ts}')
    merged_dir = merge_lora_to_base(adapter_dir, out_dir)
    ti.xcom_push(key='merged_dir', value=merged_dir)


def run_deploy(**context):
    ti = context['ti']
    merged_dir = ti.xcom_pull(key='merged_dir', task_ids='merge_lora')
    # Reads HF token from env var HF_TOKEN
    push_merged_to_hub(merged_dir, MERGED_REPO_NAME)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    dag_id='continuous_finetune',
    default_args=default_args,
    description='Ingest -> Train -> Evaluate -> Gate -> Merge -> Deploy',
    schedule_interval=None,  # set a cron string for periodic runs
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t_ingest = PythonOperator(
        task_id='ingest_preprocess',
        python_callable=run_ingest,
    )

    t_train = PythonOperator(
        task_id='train_model',
        python_callable=run_training,
    )

    t_eval = PythonOperator(
        task_id='evaluate_model',
        python_callable=run_evaluation,
    )

    t_gate = BranchPythonOperator(
        task_id='quality_gate',
        python_callable=quality_gate,
    )

    t_merge = PythonOperator(
        task_id='merge_lora',
        python_callable=run_merge,
    )

    t_deploy = PythonOperator(
        task_id='deploy_merged',
        python_callable=run_deploy,
    )

    t_stop = EmptyOperator(task_id='stop_pipeline')

    t_ingest >> t_train >> t_eval >> t_gate
    t_gate >> t_merge >> t_deploy
    t_gate >> t_stop
