"""Metrics with uncertainty, and paired comparisons between models on the same episodes.

    uv run --locked python -m ml45bench.report runs/gemma4-26b-a4b runs/nemotron-super-vl ...

Per model: success rate with a 95% Wilson interval, per-task success, steps to success,
partial progress (max dense reward / 10, grasp rate), invalid replies, tokens and latency
per call / per episode. Between models: exact McNemar test on the paired episodes and a
paired bootstrap 95% CI for the difference in success rate.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def bootstrap_mean(xs: list[float], b: int = 10000, seed: int = 0) -> list[float]:
    if not xs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(ms[int(0.025 * b)], 4), round(ms[int(0.975 * b) - 1], 4)]


def _pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else None


def summarize(results: list[dict]) -> dict:
    n = len(results)
    k = sum(r["success"] for r in results)
    per_task = defaultdict(list)
    for r in results:
        per_task[r["env"]].append(r["success"])
    calls = sum(r["calls"] for r in results)
    lat = [r["model_time_s"] / r["calls"] for r in results if r["calls"]]
    prog = [min(1.0, r["max_reward"] / 10.0) for r in results]
    s2s = [r["steps_to_success"] for r in results if r["success"]]
    return {
        "n": n, "successes": k, "success_rate": round(k / n, 4) if n else 0.0, "success_ci95": wilson(k, n),
        "per_task": {t: {"n": len(v), "successes": sum(v), "rate": round(sum(v) / len(v), 4),
                         "ci95": wilson(sum(v), len(v))} for t, v in sorted(per_task.items())},
        "steps_to_success_median": _pct(s2s, 0.5),
        "progress_mean": round(sum(prog) / n, 4) if n else 0.0, "progress_ci95": bootstrap_mean(prog),
        "grasp_rate": round(sum(r["grasped"] for r in results) / n, 4) if n else 0.0,
        "calls": calls, "invalid_reply_rate": round(sum(r["invalid_replies"] for r in results) / max(1, calls), 4),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "tokens_per_episode_mean": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in results) / max(1, n), 1),
        "tokens_per_success": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in results) / k, 1) if k else None,
        "latency_per_call_p50_s": _pct(lat, 0.5), "latency_per_call_p95_s": _pct(lat, 0.95),
        "model_time_per_episode_mean_s": round(sum(r["model_time_s"] for r in results) / max(1, n), 2),
        "time_to_success_median_s": _pct([r["model_time_s"] for r in results if r["success"]], 0.5),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def paired(a: dict[str, bool], b: dict[str, bool], boot: int = 10000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    d = [int(a[i]) - int(b[i]) for i in ids]
    return {"n_paired": len(ids), "diff": round(sum(d) / len(ids), 4) if ids else 0.0,
            "diff_ci95": bootstrap_mean(d, boot, seed), "only_first": only_a, "only_second": only_b,
            "mcnemar_p": mcnemar_exact(only_a, only_b)}


def load_run(d: Path) -> tuple[str, list[dict]]:
    res = [json.loads(p.read_text()) for p in sorted((d / "episodes").glob("*.json"))]
    return d.name, res


def main(dirs: list[str]) -> None:
    runs = [load_run(Path(d)) for d in dirs]
    table = []
    for name, res in runs:
        s = summarize(res)
        table.append({"run": name, **s})
        print(f"{name:28s} n={s['n']:3d} success {s['successes']:3d} = {s['success_rate']:.1%} "
              f"CI95 [{s['success_ci95'][0]:.1%}, {s['success_ci95'][1]:.1%}]  progress {s['progress_mean']:.3f}  "
              f"grasp {s['grasp_rate']:.0%}  invalid {s['invalid_reply_rate']:.1%}  "
              f"tok/ep {s['tokens_per_episode_mean']:.0f}  p50 {s['latency_per_call_p50_s']}s")
        for t, v in s["per_task"].items():
            print(f"    {t:18s} {v['successes']:2d}/{v['n']}")
    comps = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a = {r["id"]: r["success"] for r in runs[i][1]}
            b = {r["id"]: r["success"] for r in runs[j][1]}
            p = paired(a, b)
            comps.append({"first": runs[i][0], "second": runs[j][0], **p})
            print(f"{runs[i][0]} vs {runs[j][0]}: diff {p['diff']:+.1%} CI95 {p['diff_ci95']} "
                  f"(discordant {p['only_first']}/{p['only_second']}, McNemar p={p['mcnemar_p']})")
    out = Path("runs") / "comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"runs": table, "paired": comps}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
