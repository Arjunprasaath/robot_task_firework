"""Upload an SFT dataset to Fireworks and run LoRA supervised fine-tuning jobs.

    uv run --locked python -m ml45bench.sft upload ml45-train45 data/sft_train45/train.jsonl
    uv run --locked python -m ml45bench.sft train nemotron-ml45-sft --dataset ml45-train45 \
        --base-model accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16 --epochs 1 --lora-rank 64
    uv run --locked python -m ml45bench.sft train nemotron-ml45-5shot --dataset ml45-5shot \
        --warm-start accounts/{account}/models/nemotron-ml45-sft --epochs 3 --lora-rank 64
    uv run --locked python -m ml45bench.sft status nemotron-ml45-sft

Large files go through a signed URL (datasets/{id}:getUploadEndpoint), small ones through
datasets/{id}:upload (multipart, <150 MB). The job id is also the output model name.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from . import fireworks as fw

SMALL = 100 * 2**20


def _req(method: str, path: str, **kw) -> dict:
    r = requests.request(method, fw._acct_url(path), headers=fw._headers(), timeout=kw.pop("timeout", 60), **kw)
    if r.status_code >= 400:
        raise fw.APIError(f"{method} {path}: HTTP {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}


def upload(dataset_id: str, path: Path) -> None:
    n = sum(1 for _ in open(path))
    size = path.stat().st_size
    try:
        _req("POST", "datasets", data=json.dumps({"datasetId": dataset_id, "dataset": {
            "displayName": dataset_id, "format": "CHAT", "exampleCount": str(n), "userUploaded": {}}}))
        print(f"created dataset {dataset_id} ({n} rows, {size / 2**20:.0f} MB)")
    except fw.APIError as e:
        if "409" not in str(e) and "already exists" not in str(e).lower():
            raise
        print(f"dataset {dataset_id} exists; uploading again")
    fname = path.name
    if size < SMALL:
        with open(path, "rb") as f:
            r = requests.post(fw._acct_url(f"datasets/{dataset_id}:upload"),
                              headers={"Authorization": fw._headers()["Authorization"]},
                              files={"file": (fname, f)}, timeout=600)
        if r.status_code >= 400:
            raise fw.APIError(f"upload: HTTP {r.status_code}: {r.text[:500]}")
    else:
        urls = _req("POST", f"datasets/{dataset_id}:getUploadEndpoint",
                    data=json.dumps({"filenameToSize": {fname: str(size)}}))["filenameToSignedUrls"]
        with open(path, "rb") as f:
            r = requests.put(urls[fname], data=f, timeout=3600, headers={
                "Content-Type": "application/octet-stream", "x-goog-content-length-range": f"{size},{size}"})
        if r.status_code >= 400:
            raise fw.APIError(f"signed upload: HTTP {r.status_code}: {r.text[:500]}")
        _req("POST", f"datasets/{dataset_id}:validateUpload", data="{}")
    for _ in range(120):
        d = _req("GET", f"datasets/{dataset_id}")
        if d.get("state") == "READY":
            break
        time.sleep(5)
    print(json.dumps({k: d.get(k) for k in ("name", "state", "status", "exampleCount", "estimatedTokenCount")}))


def train(job_id: str, dataset: str, base_model: str | None, warm_start: str | None, epochs: int,
          lora_rank: int, lr: float | None, max_context: int | None) -> None:
    acct = fw.account()
    body = {"dataset": f"accounts/{acct}/datasets/{dataset}", "outputModel": f"accounts/{acct}/models/{job_id}",
            "displayName": job_id, "epochs": epochs, "loraRank": lora_rank}
    if base_model:
        body["baseModel"] = base_model
    else:
        body["warmStartFrom"] = warm_start.replace("{account}", acct)
    if lr is not None:
        body["learningRate"] = lr
    if max_context:
        body["maxContextLength"] = max_context
    j = _req("POST", "supervisedFineTuningJobs", params={"supervisedFineTuningJobId": job_id}, data=json.dumps(body))
    print(json.dumps(j, indent=1))


def status(job_id: str) -> None:
    j = _req("GET", f"supervisedFineTuningJobs/{job_id}")
    print(json.dumps({k: j.get(k) for k in ("name", "state", "status", "jobProgress", "estimatedCost",
                                             "outputModel", "createTime", "completedTime")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("dataset_id")
    u.add_argument("path", type=Path)
    t = sub.add_parser("train")
    t.add_argument("job_id")
    t.add_argument("--dataset", required=True)
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-model")
    g.add_argument("--warm-start")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lora-rank", type=int, default=64)
    t.add_argument("--lr", type=float)
    t.add_argument("--max-context", type=int)
    s = sub.add_parser("status")
    s.add_argument("job_id")
    a = ap.parse_args()
    if a.cmd == "upload":
        upload(a.dataset_id, a.path)
    elif a.cmd == "train":
        train(a.job_id, a.dataset, a.base_model, a.warm_start, a.epochs, a.lora_rank, a.lr, a.max_context)
    else:
        status(a.job_id)


if __name__ == "__main__":
    main()
