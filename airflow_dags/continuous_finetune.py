# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import sys
sys.path.extend(["/opt/airflow", "/opt/airflow/src"])  # allow importing local project modules inside containers
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.empty import EmptyOperator
import requests
import os
from huggingface_hub import HfApi

# Project modules
from src.ingest import materialize_cleaned_dataset
from src.train import train as train_pipeline
from src.metrics_eval import generate_predictions, compute_bleu, compute_bertscore
from src.merge import merge_lora_to_base
from src.save_upload import push_merged_to_hub
from src.config import QUALITY_THRESHOLDS, ADAPTERS_DIR
from src.pipeline_settings import (
    DATA_DIR as DEF_DATA_DIR,
    LORA_OUTPUT_DIR as DEF_LORA_DIR,
    MERGED_OUTPUT_DIR as DEF_MERGED_DIR,
    TRITON_MODEL_REPO as DEF_TRITON_REPO,
    TRITON_DOCKER_IMAGE as DEF_TRITON_IMAGE,
    BASE_MODEL_ID as DEF_BASE_MODEL,
    HF_MERGED_REPO as DEF_HF_REPO,
    BLEU_MIN as DEF_BLEU_MIN,
    BERTSCORE_F1_MIN as DEF_BERT_F1_MIN,
    TRITON_HTTP_URL as DEF_TRITON_URL,
    TRITON_MODEL_NAME as DEF_TRITON_MODEL,
)


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
    import os
    import torch
    from peft import PeftModel
    # Allow overriding base model via Airflow Variable; fallback to config.MODEL_ID
    model_id = Variable.get('BASE_MODEL_ID', default_var=DEF_BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    offload_dir = str(ADAPTERS_DIR.parent / 'offload')
    os.makedirs(offload_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Load dataset from disk
    from src.ingest import load_cleaned_split
    split = load_cleaned_split(cleaned_root)

    preds, refs = generate_predictions(model, tok, split['test'], limit=20)
    bleu = compute_bleu(preds, refs)
    bert = compute_bertscore(preds, refs)
    ti.xcom_push(key='bleu', value=bleu)
    ti.xcom_push(key='bertscore', value=bert)


def quality_gate(**context):
    ti = context['ti']
    bleu = ti.xcom_pull(key='bleu', task_ids='evaluate_model') or {}
    bert = ti.xcom_pull(key='bertscore', task_ids='evaluate_model') or {}
    # Read thresholds from Airflow Variables if present, else fall back to central defaults
    try:
        bleu_min = float(Variable.get('BLEU_MIN', default_var=DEF_BLEU_MIN))
    except Exception:
        bleu_min = DEF_BLEU_MIN
    try:
        berts_min = float(Variable.get('BERTSCORE_F1_MIN', default_var=DEF_BERT_F1_MIN))
    except Exception:
        berts_min = DEF_BERT_F1_MIN
    bleu_ok = bleu.get('bleu', 0) >= bleu_min
    bert_ok = bert.get('f1', 0) >= berts_min
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
    # Repo can be overridden via Airflow Variable; fallback to config.MERGED_REPO_NAME
    repo = Variable.get('HF_MERGED_REPO', default_var=DEF_HF_REPO)
    # Reads HF token from env var HF_TOKEN
    push_merged_to_hub(merged_dir, repo)


def check_hf_data_status(**context):
    """Checks if the dataset on HF has a newer commit than what we last processed."""
    api = HfApi()
    # Monitor the source dataset
    dataset_id = "moamineelhilali/python_code_instructions_18k_alpaca"
    
    try:
        info = api.dataset_info(dataset_id)
        latest_sha = info.sha
        
        # Store/retrieve last seen SHA from Airflow Variable
        var_name = f"hf_last_sha_{dataset_id.replace('/', '_')}"
        last_sha = Variable.get(var_name, default_var=None)
        
        if latest_sha != last_sha:
            print(f"New data detected! SHA changed from {last_sha} to {latest_sha}")
            Variable.set(var_name, latest_sha)
            return True
        else:
            print(f"No new data. Current SHA {latest_sha} matches last processed.")
            return False
    except Exception as e:
        print(f"Error checking HF: {e}")
        return False


def run_reload(**context):
    import requests
    # Allow overriding via Airflow Variables
    triton_url = Variable.get('TRITON_HTTP_URL', default_var=DEF_TRITON_URL)
    model_name = Variable.get('TRITON_MODEL_NAME', default_var=DEF_TRITON_MODEL)
    
    # Triton Model Control API: /v2/repository/models/{model}/load
    load_url = f"{triton_url}/v2/repository/models/{model_name}/load"
    
    print(f"Pinging Triton to reload model at: {load_url}")
    response = requests.post(load_url)
    if response.status_code == 200:
        print(f"Successfully reloaded model '{model_name}' in Triton.")
    else:
        print(f"Warning: Triton reload returned status {response.status_code}: {response.text}")
        # We don't necessarily want to fail the whole pipeline if just the reload fails,
        # but you can raise an exception here if you want it to show as 'FAILED' in Airflow.


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
    'execution_timeout': timedelta(minutes=30),
}

with DAG(
    dag_id='continuous_finetune',
    default_args=default_args,
    description='Ingest -> Train -> Evaluate -> Gate -> Merge -> Deploy',
    schedule_interval="@hourly", # Check every hour for new data
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t_sensor = PythonSensor(
        task_id='check_for_new_data',
        python_callable=check_hf_data_status,
        timeout=60 * 5,  # 5 minutes
        mode='reschedule', # Release slot while waiting
        poke_interval=60 * 60, # Only poke once per hour (matches schedule)
    )

    t_ingest = PythonOperator(
        task_id='ingest_preprocess',
        python_callable=run_ingest,
    )

    t_train = PythonOperator(
        task_id='train_model',
        python_callable=run_training,
        execution_timeout=timedelta(hours=12),
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

    t_reload = PythonOperator(
        task_id='reload_triton',
        python_callable=run_reload,
    )

    t_stop = EmptyOperator(task_id='stop_pipeline')

    t_sensor >> t_ingest >> t_train >> t_eval >> t_gate
    t_gate >> t_merge >> t_deploy >> t_reload
    t_gate >> t_stop
