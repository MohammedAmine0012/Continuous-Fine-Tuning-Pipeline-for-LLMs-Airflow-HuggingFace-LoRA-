import json
import base64
import argparse
import requests

# Simple Triton HTTP client for BYTES I/O model

def infer(url: str, model: str, prompt: str):
    payload = {
        "inputs": [
            {
                "name": "prompt",
                "datatype": "BYTES",
                "shape": [1, 1],
                "data": [prompt]
            }
        ],
        "outputs": [
            {"name": "text"}
        ]
    }
    r = requests.post(f"{url}/v2/models/{model}/infer", json=payload, timeout=300)
    if not r.ok:
        raise RuntimeError(f"Inference failed: {r.status_code} {r.text}")
    out = r.json()
    data = out["outputs"][0]["data"][0]
    return data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="Triton HTTP endpoint")
    ap.add_argument("--model", default="deepseek_merged", help="Model name in Triton")
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    print(infer(args.url, args.model, args.prompt))
