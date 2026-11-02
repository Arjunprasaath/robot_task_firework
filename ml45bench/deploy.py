"""Bring benchmark models up / down as Fireworks on-demand deployments.

    uv run --locked python -m ml45bench.deploy up glimmer-30b nemotron-super-vl qwen3p8-27b gemma4-26b-a4b
    uv run --locked python -m ml45bench.deploy status
    uv run --locked python -m ml45bench.deploy probe gemma4-26b-a4b     # one real benchmark request
    uv run --locked python -m ml45bench.deploy up nemotron-ml45-sft      # fine-tuned, live-merged
    uv run --locked python -m ml45bench.deploy down ml45-gemma4-26b-a4b

Each model uses the validated Fireworks deployment shape pinned in models.json, min 0 / max 1
replica, and scales to zero after 15 idle minutes (billed per GPU-second while up).
"""
from __future__ import annotations

import json
import sys
import time

from . import fireworks as fw
from .run import MODELS, model_address


def _cfgs():
    return json.loads(MODELS.read_text())


def up(keys):
    cfgs = _cfgs()
    have = {d["name"].split("/")[-1]: d for d in fw.list_deployments()}
    for k in keys:
        c = cfgs[k]
        if c["deployment"] in have:
            print(f"{k}: {c['deployment']} exists ({have[c['deployment']].get('state')})")
            continue
        d = fw.create_deployment(c["deployment"], c.get("serve_model", c["base_model"]).replace("{account}", fw.account()), shape=c.get("shape", "default"),
                                 enable_addons=c.get("enable_addons", False))
        print(f"{k}: created {d.get('name')} state={d.get('state')} shape={d.get('deploymentShape')} "
              f"accel={d.get('acceleratorCount')}x{d.get('acceleratorType')} precision={d.get('precision')}")


def status():
    for d in fw.list_deployments():
        print(d["name"].split("/")[-1], d.get("state"), d.get("baseModel"), f"replicas={d.get('replicaCount')}",
              f"{d.get('acceleratorCount')}x{d.get('acceleratorType')}", d.get("precision"),
              (d.get("status") or {}).get("message", "")[:120])


def probe(key):
    """Send one real benchmark request (first frame of the first held-out episode)."""
    from . import sim, spec
    from .manifest import load
    from .run import _request
    e = load()["episodes"][0]
    ep = sim.Episode(e["task_index"], e["seed"])
    jpg, ee, closed = ep.observe()
    cfg = _cfgs()[key]
    body = _request(model_address(cfg, fw.account()), cfg, spec.messages(e["task_index"], ee, closed, sim.data_url(jpg)))
    t = time.time()
    resp, dt, att = fw.chat(body, retries=40)
    msg = resp["choices"][0]["message"]
    print(json.dumps({"reply": msg.get("content"), "parsed": spec.parse_answer(msg.get("content") or ""),
                      "reasoning_chars": len(msg.get("reasoning_content") or ""), "usage": resp.get("usage"),
                      "latency_s": round(dt, 3), "attempts": att, "total_wait_s": round(time.time() - t, 1)}, indent=1))


def main(argv):
    cmd, rest = argv[0], argv[1:]
    if cmd == "up":
        up(rest)
    elif cmd == "status":
        status()
    elif cmd == "probe":
        probe(rest[0])
    elif cmd == "down":
        for d in rest:
            fw.delete_deployment(d)
            print("deleted", d)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
