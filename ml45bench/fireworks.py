"""Minimal Fireworks REST client: chat completions + on-demand deployments.

The API key is read from the environment variable FIREWORK_API, or from a `.env` file
(KEY=VALUE lines) in the current directory or the repo root. It is never logged.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests

API = "https://api.fireworks.ai"
KEY_NAME = "FIREWORK_API"


def api_key() -> str:
    if os.environ.get(KEY_NAME):
        return os.environ[KEY_NAME]
    for d in (Path.cwd(), Path(__file__).resolve().parent.parent):
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() == KEY_NAME and v.strip():
                    return v.strip().strip('"').strip("'")
    raise SystemExit(f"{KEY_NAME} not set (env var or .env)")


def _headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def account() -> str:
    acc = os.environ.get("FIREWORKS_ACCOUNT")
    if acc:
        return acc
    r = requests.get(f"{API}/v1/accounts", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()["accounts"][0]["name"].split("/")[-1]


class APIError(RuntimeError):
    pass


def chat(body: dict, retries: int = 12, timeout: float = 120.0) -> tuple[dict, float, int]:
    """POST the request unchanged until it succeeds. Retries only transport errors,
    429 and 5xx (e.g. 503 while a deployment scales up); the request is never altered.
    Returns (response json, latency of the successful attempt in s, attempts)."""
    last = None
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{API}/inference/v1/chat/completions", headers=_headers(),
                              data=json.dumps(body), timeout=timeout)
            dt = time.perf_counter() - t0
            if r.status_code == 200:
                return r.json(), dt, attempt
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 409, 425, 429) and r.status_code < 500:
                raise APIError(last)
        except requests.RequestException as exc:
            last = repr(exc)
        time.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)) * (0.5 + random.random()))
    raise APIError(f"gave up after {retries} attempts: {last}")


# ------------------------------------------------------------------ deployments
def _acct_url(path: str) -> str:
    return f"{API}/v1/accounts/{account()}/{path}"


def create_deployment(deployment_id: str, base_model: str, shape: str = "default", enable_addons: bool = False,
                      max_replicas: int = 1, scale_to_zero: str = "900s") -> dict:
    body = {"baseModel": base_model, "deploymentShape": shape, "minReplicaCount": 0,
            "maxReplicaCount": max_replicas, "enableAddons": enable_addons,
            "displayName": deployment_id,
            "autoscalingPolicy": {"scaleToZeroWindow": scale_to_zero}}
    r = requests.post(_acct_url("deployments"), params={"deploymentId": deployment_id},
                      headers=_headers(), data=json.dumps(body), timeout=60)
    if r.status_code >= 400:
        raise APIError(f"create deployment: HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def get_deployment(deployment_id: str) -> dict:
    r = requests.get(_acct_url(f"deployments/{deployment_id}"), headers=_headers(), timeout=30)
    if r.status_code >= 400:
        raise APIError(f"get deployment: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def list_deployments() -> list[dict]:
    r = requests.get(_acct_url("deployments"), headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("deployments", [])


def delete_deployment(deployment_id: str) -> None:
    r = requests.delete(_acct_url(f"deployments/{deployment_id}"), params={"ignoreChecks": "true"},
                        headers=_headers(), timeout=60)
    if r.status_code >= 400:
        raise APIError(f"delete deployment: HTTP {r.status_code}: {r.text[:300]}")


def load_lora(lora_model: str, deployment_id: str) -> dict:
    body = {"model": lora_model, "deployment": f"accounts/{account()}/deployments/{deployment_id}"}
    r = requests.post(_acct_url("deployedModels"), headers=_headers(), data=json.dumps(body), timeout=60)
    if r.status_code >= 400:
        raise APIError(f"load lora: HTTP {r.status_code}: {r.text[:500]}")
    return r.json()
