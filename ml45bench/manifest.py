"""Freeze the held-out episode set into benchmark/manifest.json.

For every (test task, seed) it records the initial object/goal position and the sha256 of
the first JPEG the model will see; the runner checks both before every episode, so a
different MetaWorld/MuJoCo/Pillow build that changes the episodes is detected instead of
silently producing different numbers. It also runs MetaWorld's scripted expert (raw and
through our quantized action codec) on every episode: a reference ceiling and a check that
every episode is solvable within MAX_STEPS.

    uv run --locked python -m ml45bench.manifest            # build
    uv run --locked python -m ml45bench.manifest --check    # rebuild in memory, compare
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np

from . import sim, spec

OUT = Path(__file__).resolve().parent.parent / "benchmark" / "manifest.json"


def expert_policy(env_name: str):
    from metaworld import policies
    name = "Sawyer" + "".join(p.capitalize() for p in env_name.removesuffix("-v3").split("-")) + "V3Policy"
    return getattr(policies, name)()


def run_expert(task_index: int, seed: int, quantized: bool) -> dict:
    ep = sim.Episode(task_index, seed)
    pol = expert_policy(spec.TEST_TASKS[task_index])
    while not ep.done:
        a = np.clip(pol.get_action(ep.obs), -1, 1)
        if quantized:
            ep.step([spec.quantize(a)])
        else:
            _raw_step(ep, a)
    out = {"success": ep.success_step is not None, "steps": ep.success_step}
    ep.close()
    return out


def _raw_step(ep: sim.Episode, a) -> None:
    ep.obs, _, _, _, info = ep.env.step(a)
    ep.t += 1
    if info.get("success", 0.0) > 0 and ep.success_step is None:
        ep.success_step = ep.t


def versions() -> dict:
    return {p: md.version(p) for p in ("metaworld", "mujoco", "gymnasium", "numpy", "pillow")} | {
        "python": sys.version.split()[0], "platform": sys.platform}


def build(with_experts: bool = True) -> dict:
    episodes = []
    for ti, env_name in spec.TEST_TASKS.items():
        for k in range(spec.SEEDS_PER_TASK):
            seed = spec.SEED_START + k
            ep = sim.Episode(ti, seed)
            jpg, ee, closed = ep.observe()
            rec = {"id": f"{env_name}:{seed}", "task_index": ti, "env": env_name, "seed": seed,
                   **sim.scene(ep.obs), "ee0": ee, "image0_sha": sim.sha(jpg)}
            ep.close()
            if with_experts:
                rec["expert"] = run_expert(ti, seed, quantized=False)
                rec["expert_quantized"] = run_expert(ti, seed, quantized=True)
            episodes.append(rec)
            print(rec["id"], rec["obj"], rec["goal"], rec.get("expert"), rec.get("expert_quantized"), flush=True)
    scenes = {(e["env"], tuple(e["obj"]), tuple(e["goal"])) for e in episodes}
    assert len(scenes) == len(episodes), "two held-out episodes share the same scene"
    return {"bench_version": spec.BENCH_VERSION, "spec_fingerprint": spec.spec_fingerprint(),
            "built_with": versions(), "max_steps": spec.MAX_STEPS, "chunk": spec.CHUNK,
            "system_prompt": spec.system_prompt(), "grammar": spec.GRAMMAR, "decoding": spec.DECODING,
            "goal_names": {spec.TEST_TASKS[t]: spec.GOAL_NAMES[t] for t in spec.TEST_TASKS},
            "episodes": episodes}


def load() -> dict:
    m = json.loads(OUT.read_text())
    if m["spec_fingerprint"] != spec.spec_fingerprint():
        raise SystemExit(f"spec.py changed since the manifest was frozen "
                         f"({m['spec_fingerprint']} != {spec.spec_fingerprint()}); rebuild + bump BENCH_VERSION")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare to the frozen manifest")
    args = ap.parse_args()
    if args.check:
        frozen = load()
        new = build(with_experts=False)
        bad = [(a["id"], k) for a, b in zip(frozen["episodes"], new["episodes"])
               for k in ("obj", "goal", "ee0", "image0_sha") if a[k] != b[k]]
        print("MATCH" if not bad else f"MISMATCH {bad[:10]}")
        sys.exit(1 if bad else 0)
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=1))
    ex = [e["expert"]["success"] for e in m["episodes"]]
    eq = [e["expert_quantized"]["success"] for e in m["episodes"]]
    print(f"wrote {OUT}: {len(ex)} episodes; scripted expert {sum(ex)}/{len(ex)}, quantized {sum(eq)}/{len(eq)}")


if __name__ == "__main__":
    main()
