"""Run one model on the frozen 100 held-out episodes.

    uv run --locked python -m ml45bench.run --model qwen3p8-27b --concurrency 20
    uv run --locked python -m ml45bench.run --model expert           # offline pipeline check
    uv run --locked python -m ml45bench.run --replay runs/<run>      # re-simulate a transcript

One process. The main thread owns every simulator and renderer (MuJoCo/OpenGL must stay on
one thread); a thread pool only waits on Fireworks HTTP calls. Up to --concurrency episodes
are in flight at once, which is what sets the speed (the sim is ~10 ms per call, a reply ~2-3 s).

Every model gets the same episodes, prompt, image, grammar and decoding (spec.py); the only
per-model fields are its Fireworks address and the reasoning switch (models.json). Each
episode writes a transcript (one line per call: image hash, prompt, raw reply, parsed actions,
latency, tokens) and a result. Re-running resumes; `--replay` re-simulates the recorded
actions without the API and checks every image hash and outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import sim, spec

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
    return {"model": addr, "messages": msgs, **spec.DECODING, **cfg.get("extra", {})}


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _fname(eid: str) -> str:
    return eid.replace(":", "_")


class Runner:
    """State of one in-flight episode (lives on the main thread)."""

    def __init__(self, e: dict, out: Path, video: bool):
        self.e, self.out = e, out
        self.ep = sim.Episode(e["task_index"], e["seed"])
        got = sim.scene(self.ep.obs)
        if got["obj"] != e["obj"] or got["goal"] != e["goal"]:
            raise RuntimeError(f"{e['id']}: scene differs from manifest {got}")
        self.calls = open(out / "calls" / f"{_fname(e['id'])}.jsonl", "w")
        self.writer = None
        if video:
            import imageio.v2 as imageio
            (out / "videos").mkdir(exist_ok=True)
            self.writer = imageio.get_writer(out / "videos" / f"{_fname(e['id'])}.mp4", fps=10, macro_block_size=1)
        self.n = self.invalid = self.p_tok = self.c_tok = 0
        self.model_time = 0.0
        self.lat: list[float] = []
        self.t0 = time.perf_counter()
        self.pending = None  # (msgs, img_sha, ee, closed)

    def next_obs(self):
        rgb = sim.frame(self.ep.env, self.ep.renderer)
        if self.writer is not None:
            self.writer.append_data(rgb)
        jpg = sim.jpeg(rgb)
        img_sha = sim.sha(jpg)
        if self.n == 0 and img_sha != self.e["image0_sha"]:
            raise RuntimeError(f"{self.e['id']}: first image differs from manifest (renderer/version drift)")
        ee, closed = sim.proprio(self.ep.obs)
        msgs = spec.messages(self.e["task_index"], ee, closed, sim.data_url(jpg))
        self.pending = (msgs, img_sha, ee, closed)
        return msgs

    def apply(self, text: str, acts, valid: bool, dt: float, attempts: int, resp: dict) -> None:
        msgs, img_sha, ee, closed = self.pending
        usage = resp.get("usage") or {}
        choice = (resp.get("choices") or [{}])[0]
        self.n += 1
        self.invalid += 0 if valid else 1
        self.model_time += dt
        self.lat.append(dt)
        self.p_tok += int(usage.get("prompt_tokens") or 0)
        self.c_tok += int(usage.get("completion_tokens") or 0)
        self.calls.write(json.dumps({
            "call": self.n, "t": self.ep.t, "image_sha": img_sha, "ee": ee, "closed": closed,
            "user_text": msgs[1]["content"][0]["text"], "reply": text, "actions": acts, "valid": valid,
            "latency_s": round(dt, 4), "attempts": attempts, "usage": usage,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len((choice.get("message") or {}).get("reasoning_content") or ""),
        }) + "\n")
        self.ep.step(acts)

    def finish(self) -> dict:
        ep, e = self.ep, self.e
        self.calls.close()
        if self.writer is not None:
            self.writer.close()
        succ = ep.success_step is not None
        res = {
            "id": e["id"], "env": e["env"], "task_index": e["task_index"], "seed": e["seed"],
            "success": succ, "steps": ep.t, "steps_to_success": ep.success_step,
            "calls": self.n, "invalid_replies": self.invalid,
            "prompt_tokens": self.p_tok, "completion_tokens": self.c_tok,
            "model_time_s": round(self.model_time, 3), "wall_time_s": round(time.perf_counter() - self.t0, 3),
            "latency_p50_s": round(sorted(self.lat)[len(self.lat) // 2], 4) if self.lat else None,
            "max_reward": round(ep.max_reward, 4), "grasped": ep.grasped,
            "min_obj_to_target": round(ep.min_obj_to_target, 4),
        }
        ep.close()
        _write_json(self.out / "episodes" / f"{_fname(e['id'])}.json", res)
        print(f"[{e['id']}] success={succ} steps={ep.t} calls={self.n} invalid={self.invalid} "
              f"tok={self.p_tok}+{self.c_tok} model={self.model_time:.0f}s", flush=True)
        return res


def _call_model(addr: str, cfg: dict, msgs: list[dict]):
    from . import fireworks
    resp, dt, attempts = fireworks.chat(_request(addr, cfg, msgs))
    text = resp["choices"][0]["message"].get("content") or ""
    acts, valid = spec.parse_answer(text)
    return text, acts, valid, dt, attempts, resp


def run_all(eps: list[dict], cfg: dict, addr: str, out: Path, concurrency: int, video: bool) -> list[dict]:
    results = []
    todo = []
    for e in eps:
        p = out / "episodes" / f"{_fname(e['id'])}.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
        else:
            todo.append(e)
    expert = cfg.get("policy") == "expert"
    pool = ThreadPoolExecutor(max_workers=concurrency)
    inflight = {}  # future -> Runner
    queue = list(todo)

    def submit(r: Runner):
        msgs = r.next_obs()
        if expert:
            from .manifest import expert_policy
            pol = expert_policy(r.e["env"])
            a = spec.quantize(np.clip(pol.get_action(r.ep.obs), -1, 1))
            fut = pool.submit(lambda: (",".join(map(str, a)), [a], True, 0.0, 1, {}))
        else:
            fut = pool.submit(_call_model, addr, cfg, msgs)
        inflight[fut] = r

    while queue or inflight:
        while queue and len(inflight) < concurrency:
            submit(Runner(queue.pop(0), out, video))
        done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for fut in done:
            r = inflight.pop(fut)
            r.apply(*fut.result())
            if r.ep.done:
                results.append(r.finish())
            else:
                submit(r)
    pool.shutdown()
    return results


def replay(run_dir: Path) -> None:
    """Re-simulate recorded actions; every image hash and every outcome must match."""
    from .manifest import load
    man = {e["id"]: e for e in load()["episodes"]}
    bad = 0
    for res_path in sorted((run_dir / "episodes").glob("*.json")):
        res = json.loads(res_path.read_text())
        e = man[res["id"]]
        ep = sim.Episode(e["task_index"], e["seed"])
        ok = True
        for line in open(run_dir / "calls" / res_path.name.replace(".json", ".jsonl")):
            c = json.loads(line)
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
    ap.add_argument("--concurrency", type=int, default=20, help="episodes in flight (parallel API calls)")
    ap.add_argument("--limit", type=int, default=0, help="first N seeds per task (smoke test)")
    ap.add_argument("--video", action="store_true", help="stream the model's input frames to mp4")
    ap.add_argument("--replay", type=Path)
    args = ap.parse_args()
    if args.replay:
        return replay(args.replay)

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
    out = ROOT / "runs" / (args.run_name or args.model)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "calls").mkdir(exist_ok=True)
    meta = {"model_key": args.model, "address": addr, "config": cfg,
            "request_template": {k: v for k, v in _request(addr, cfg, []).items() if k != "messages"},
            "bench_version": man["bench_version"], "spec_fingerprint": man["spec_fingerprint"],
            "manifest_sha": hashlib.sha256((ROOT / "benchmark" / "manifest.json").read_bytes()).hexdigest()[:16],
            "code_commit": _git_commit()}
    meta_path = out / "run.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        keys = ("address", "request_template", "spec_fingerprint", "manifest_sha")
        if any(old.get(k) != meta[k] for k in keys):
            raise SystemExit(f"{out} was started with a different config; use another --run-name")
    else:
        meta["started"] = datetime.now(timezone.utc).isoformat()
        _write_json(meta_path, meta)

    t0 = time.time()
    results = run_all(eps, cfg, addr, out, args.concurrency, args.video)
    from .report import summarize
    s = summarize(results)
    s["wall_clock_s"] = round(time.time() - t0, 1)
    _write_json(out / "summary.json", s)
    print(json.dumps({k: s[k] for k in ("n", "successes", "success_rate", "success_ci95")}))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
