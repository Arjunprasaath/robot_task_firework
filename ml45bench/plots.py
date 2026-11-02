"""Figures for the README from finished runs (matplotlib is not a benchmark dependency).

    uv run --locked --with matplotlib python -m ml45bench.plots runs/glimmer-30b runs/nemotron-super-vl

Writes docs/figures/success.png, progress_by_task.png and efficiency.png.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .report import bootstrap_mean, load_run, summarize  # noqa: E402
from .run import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


SHORT = {"glimmer-30b": "Glimmer 30B", "nemotron-super-vl": "Nemotron base",
         "nemotron-ml45-sft": "Nemotron + ML45 SFT", "nemotron-ml45-5shot": "Nemotron + ML45 SFT + 5-shot"}


def _label(run_name: str) -> str:
    import json
    cfg = json.loads(MODELS.read_text()).get(run_name, {})
    return cfg.get("display", run_name)


def _short(run_name: str) -> str:
    return SHORT.get(run_name, run_name)


def progress_by_task(runs) -> Path:
    tasks = sorted({r["env"] for _, res in runs for r in res})
    rows = ["all 100 episodes"] + [t.replace("-v3", "") for t in tasks]
    fig, ax = plt.subplots(figsize=(8, 3.2 + 0.7 * len(runs)))
    h = 0.8 / len(runs)
    for k, (name, res) in enumerate(runs):
        by = defaultdict(list)
        for r in res:
            by[r["env"]].append(min(1.0, r["max_reward"] / 10.0))
        groups = [[x for t in tasks for x in by[t]]] + [by[t] for t in tasks]
        means = [sum(g) / len(g) for g in groups]
        cis = [bootstrap_mean(g) for g in groups]
        ys = [i + (k - (len(runs) - 1) / 2) * h for i in range(len(rows))]
        err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
        ax.barh(ys, means, height=h * 0.9, color=SERIES[k], label=_short(name), zorder=2)
        ax.errorbar(means, ys, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.4, ax.get_xlim()[1]))
    ax.set_xlabel("partial progress (max dense reward / 10), mean with 95% bootstrap CI")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Partial progress on the 5 held-out ML45 tasks (20 seeds each)", loc="left", fontsize=10.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    p = OUT / "progress_by_task.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def success(runs) -> Path:
    sums = [(n, summarize(res)) for n, res in runs]
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.55 * len(runs)))
    for k, (n, s) in enumerate(sums):
        lo, hi = s["success_ci95"]
        ax.plot([lo * 100, hi * 100], [k, k], color=SERIES[k], lw=2, solid_capstyle="round", zorder=2)
        ax.plot([s["success_rate"] * 100], [k], "o", ms=8, color=SERIES[k], mec=SURFACE, mew=2, zorder=3)
        ax.text(hi * 100 + 0.4, k, f"{s['successes']}/{s['n']}  (95% CI {lo:.1%}–{hi:.1%})", va="center", fontsize=9)
    ax.set_yticks(range(len(sums)), [_short(n) for n, _ in sums])
    ax.invert_yaxis()
    ax.set_xlim(0, max(20, max(s["success_ci95"][1] for _, s in sums) * 100 + 32))
    ax.set_ylim(len(sums) - 0.5, -0.5)
    ax.set_xlabel("success rate on 100 held-out episodes (%), with 95% Wilson interval")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_title("Task success on the 5 held-out ML45 tasks", loc="left", fontsize=10.5)
    fig.tight_layout()
    p = OUT / "success.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def efficiency(runs) -> Path:
    sums = [(n, summarize(res)) for n, res in runs]
    panels = [("tokens per episode (thousands)", lambda s: s["tokens_per_episode_mean"] / 1000, "{:.0f}k"),
              ("latency per call, p50 (s)", lambda s: s["latency_per_call_p50_s"], "{:.2f} s"),
              ("latency per call, p95 (s)", lambda s: s["latency_per_call_p95_s"], "{:.2f} s")]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.3 * len(runs) * len(panels), 3.4))
    for ax, (title, f, fmt) in zip(axes, panels):
        vals = [f(s) for _, s in sums]
        xs = range(len(sums))
        ax.bar(xs, vals, width=0.6, color=SERIES[:len(sums)], zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(list(xs), [_short(n).replace(" + ", "\n+ ").replace("Nemotron base", "Nemotron\nbase").replace("Glimmer 30B", "Glimmer\n30B") for n, _ in sums],
                      fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    p = OUT / "efficiency.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main(dirs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load_run(Path(d)) for d in dirs]
    for p in (success(runs), progress_by_task(runs), efficiency(runs)):
        print("->", p)


if __name__ == "__main__":
    main(sys.argv[1:])
