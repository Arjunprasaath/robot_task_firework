"""Run one model on the frozen 100 held-out episodes.

    uv run --locked python -m ml45bench.run --model gemma4-26b-a4b --workers 16
    uv run --locked python -m ml45bench.run --model expert          # offline pipeline check
    uv run --locked python -m ml45bench.run --replay runs/<run>      # re-simulate a transcript

Every model gets the same episodes, prompt, image, grammar and decoding (spec.py); the
only per-model fields are its Fireworks address and the reasoning switch (models.json).
Each episode writes a transcript (one line per model call: image hash, prompt, raw reply,
parsed actions, latency, tokens) and a result. Re-running resumes; `--replay` re-simulates
the recorded actions without the API and checks every image hash and the outcome, which
proves the numbers came from exactly these episodes and replies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "benchmark" / "models.json"


def _git_commit() -> str:
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "ml45bench", "benchmark"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def model_address(cfg: dict, acct: str) -> str:
    dep = f"accounts/{acct}/deployments/{cfg['deployment']}"
    target = cfg.get("serve_model", cfg["base_model"]).replace("{account}", acct)
    return f"{target}#{dep}"


def _request(addr: str, cfg: dict, msgs: list[dict]) -> dict:
    from . import spec
    return {"model": addr, "messages": msgs, **spec.DECODING, **cfg.get("extra", {})}


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_episode(job: dict) -> dict:
    from . import sim, spec
    e, cfg, out = job["episode"], job["cfg"], Path(job["out"])
    calls_path = out / "calls" / f"{e['id'].replace(':', '_')}.jsonl"
    res_path = out / "episodes" / f"{e['id'].replace(':', '_')}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    ep = sim.Episode(e["task_index"], e["seed"])
    got = sim.scene(ep.obs)
    if got["obj"] != e["obj"] or got["goal"] != e["goal"]:
        raise RuntimeError(f"{e['id']}: scene differs from manifest {got} vs {e}")
    expert = None
    if cfg.get("policy") == "expert":
        from .manifest import expert_policy
        import numpy as np
        expert = expert_policy(e["env"])

    t_start = time.perf_counter()
    model_time = 0.0
    n_calls = n_invalid = p_tok = c_tok = 0
    lat = []
    with open(calls_path, "w") as fh:
        while not ep.done:
            jpg, ee, closed = ep.observe(keep_frame=job["video"])
            img_sha = sim.sha(jpg)
            if n_calls == 0 and img_sha != e["image0_sha"]:
                raise RuntimeError(f"{e['id']}: first image differs from manifest (renderer/version drift)")
            msgs = spec.messages(e["task_index"], ee, closed, sim.data_url(jpg))
            if expert is not None:
                reply = spec.quantize(np.clip(expert.get_action(ep.obs), -1, 1))
                text = ",".join(map(str, reply))
                acts, valid = [reply], True
                resp, dt, attempts, usage = {}, 0.0, 1, {}
            else:
                from . import fireworks
                body = _request(job["addr"], cfg, msgs)
                resp, dt, attempts = fireworks.chat(body)
                choice = resp["choices"][0]
                text = choice["message"].get("content") or ""
                acts, valid = spec.parse_answer(text)
                usage = resp.get("usage") or {}
            n_calls += 1
            n_invalid += 0 if valid else 1
            model_time += dt
            lat.append(dt)
            p_tok += int(usage.get("prompt_tokens") or 0)
            c_tok += int(usage.get("completion_tokens") or 0)
            fh.write(json.dumps({
                "call": n_calls, "t": ep.t, "image_sha": img_sha, "ee": ee, "closed": closed,
                "user_text": msgs[1]["content"][0]["text"], "reply": text, "actions": acts,
                "valid": valid, "latency_s": round(dt, 4), "attempts": attempts, "usage": usage,
                "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
                "reasoning_chars": len((resp.get("choices") or [{}])[0].get("message", {}).get("reasoning_content") or ""),
            }) + "\n")
            fh.flush()
            ep.step(acts)
    wall = time.perf_counter() - t_start
    succ = ep.success_step is not None
    result = {
        "id": e["id"], "env": e["env"], "task_index": e["task_index"], "seed": e["seed"],
        "success": succ, "steps": ep.t, "steps_to_success": ep.success_step,
        "calls": n_calls, "invalid_replies": n_invalid,
        "prompt_tokens": p_tok, "completion_tokens": c_tok,
        "model_time_s": round(model_time, 3), "wall_time_s": round(wall, 3),
        "latency_p50_s": round(sorted(lat)[len(lat) // 2], 4) if lat else None,
        # partial progress (MetaWorld dense reward is in [0, 10])
        "max_reward": round(ep.max_reward, 4), "grasped": ep.grasped,
        "min_obj_to_target": round(ep.min_obj_to_target, 4),
    }
    if job["video"] and ep.frames:
        import imageio.v2 as imageio
        (out / "videos").mkdir(exist_ok=True)
        imageio.mimsave(out / "videos" / f"{e['id'].replace(':', '_')}.mp4", ep.frames, fps=10)
    ep.close()
    _write_json(res_path, result)
    print(f"[{e['id']}] success={succ} steps={ep.t} calls={n_calls} invalid={n_invalid} "
          f"tok={p_tok}+{c_tok} model={model_time:.0f}s", flush=True)
    return result


def replay(run_dir: Path) -> None:
    """Re-simulate recorded actions; every image hash and every outcome must match."""
    from . import sim
    from .manifest import load
    man = {e["id"]: e for e in load()["episodes"]}
    bad = 0
    for res_path in sorted((run_dir / "episodes").glob("*.json")):
        res = json.loads(res_path.read_text())
        e = man[res["id"]]
        ep = sim.Episode(e["task_index"], e["seed"])
        calls = [json.loads(line) for line in open(run_dir / "calls" / res_path.name.replace(".json", ".jsonl"))]
        ok = True
        for c in calls:
            jpg, _, _ = ep.observe()
            if sim.sha(jpg) != c["image_sha"]:
                ok = False
                break
            ep.step(c["actions"])
        ok = ok and (ep.success_step is not None) == res["success"] and ep.t == res["steps"]
        bad += not ok
        print(f"{res['id']}: {'OK' if ok else 'MISMATCH'}")
        ep.close()
    print("REPLAY MATCH" if not bad else f"REPLAY MISMATCH in {bad} episodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="key in benchmark/models.json, or 'expert'")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--workers", type=int, default=16, help="episodes in parallel (one process each)")
    ap.add_argument("--limit", type=int, default=0, help="first N episodes per task (smoke test)")
    ap.add_argument("--video", action="store_true", help="save the model's input frames as mp4")
    ap.add_argument("--replay", type=Path)
    args = ap.parse_args()
    if args.replay:
        return replay(args.replay)

    from . import spec
    from .manifest import load
    man = load()
    if args.model == "expert":
        cfg, addr = {"policy": "expert"}, "scripted-expert"
    else:
        from . import fireworks
        cfg = json.loads(MODELS.read_text())[args.model]
        addr = model_address(cfg, fireworks.account())
    eps = man["episodes"]
    if args.limit:
        eps = [e for e in eps if e["seed"] < spec.SEED_START + args.limit]
    name = args.run_name or f"{args.model}"
    out = ROOT / "runs" / name
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "calls").mkdir(exist_ok=True)
    meta_path = out / "run.json"
    meta = {"model_key": args.model, "address": addr, "config": cfg,
            "request_template": {k: v for k, v in _request(addr, cfg, []).items() if k != "messages"},
            "bench_version": man["bench_version"], "spec_fingerprint": man["spec_fingerprint"],
            "manifest_sha": hashlib.sha256((ROOT / "benchmark" / "manifest.json").read_bytes()).hexdigest()[:16],
            "code_commit": _git_commit(), "episodes": len(eps)}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        same = {k: old.get(k) for k in ("address", "request_template", "spec_fingerprint", "manifest_sha")}
        if same != {k: meta[k] for k in same}:
            raise SystemExit(f"{out} was started with a different config; use another --run-name")
    else:
        meta["started"] = datetime.now(timezone.utc).isoformat()
        _write_json(meta_path, meta)

    jobs = [{"episode": e, "cfg": cfg, "addr": addr, "out": str(out), "video": args.video} for e in eps]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(run_episode, jobs, chunksize=1)
    from .report import summarize
    s = summarize(results)
    s["wall_clock_s"] = round(time.time() - t0, 1)
    _write_json(out / "summary.json", s)
    print(json.dumps({k: s[k] for k in ("n", "successes", "success_rate", "success_ci95")}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
