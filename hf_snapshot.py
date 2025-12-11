import os
from huggingface_hub import snapshot_download
repo = os.environ.get("HF_MODEL_ID", "Moamineelhilali/deepseek-stackoverflow-merged")
out = snapshot_download(repo_id=repo, allow_patterns="*")
print("LOCAL_MODEL:", out)
