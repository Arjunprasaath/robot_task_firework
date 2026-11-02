"""Render episodes from recorded runs to mp4 (no API: re-simulates the logged actions).

    uv run --locked python -m ml45bench.video --episodes box-close-v3:1004 door-lock-v3:1012 \
        --runs runs/glimmer-30b runs/nemotron-super-vl runs/nemotron-ml45-5shot --out videos

One frame per env step, from the model's camera (the same image it was shown, at 480 px).
Every model input image is checked against the logged hash, so a video is the episode the
model actually played. Writes <out>/<run>__<episode>.mp4 and <out>/side_by_side__<episode>.mp4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import sim
from .manifest import load
from .plots import SHORT

FPS = 20


def _font(size: int):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(rgb: np.ndarray, top: str, bottom: str, ok: bool | None) -> np.ndarray:
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((8, 5), top, fill=(255, 255, 255), font=_font(18))
    color = (255, 255, 255) if ok is None else ((40, 200, 90) if ok else (230, 80, 70))
    d.rectangle([0, img.height - 30, img.width, img.height], fill=(0, 0, 0))
    d.text((8, img.height - 25), bottom, fill=color, font=_font(18))
    return np.asarray(img)


def frames(run_dir: Path, eid: str) -> list[np.ndarray]:
    e = {x["id"]: x for x in load()["episodes"]}[eid]
    fname = eid.replace(":", "_")
    res = json.loads((run_dir / "episodes" / f"{fname}.json").read_text())
    name = SHORT.get(run_dir.name, run_dir.name)
    ep = sim.Episode(e["task_index"], e["seed"])
    out = []

    def grab():
        status = f"step {ep.t}" + ("   SUCCESS" if ep.success_step is not None else "")
        ok = True if ep.success_step is not None else (False if ep.done else None)
        out.append(_label(sim.frame(ep.env, ep.renderer), f"{name}  |  {eid}", status, ok))

    grab()
    for line in open(run_dir / "calls" / f"{fname}.jsonl"):
        c = json.loads(line)
        if sim.sha(sim.jpeg(sim.frame(ep.env, ep.renderer))) != c["image_sha"]:
            raise RuntimeError(f"{run_dir.name} {eid} call {c['call']}: image differs from log")
        for a in c["actions"]:
            if ep.done:
                break
            ep.step([a])
            grab()
    if (ep.success_step is not None) != res["success"]:
        raise RuntimeError(f"{run_dir.name} {eid}: outcome differs from log")
    ep.close()
    return out + [out[-1]] * FPS  # hold the last frame for a second


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("videos"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for eid in a.episodes:
        clips = []
        for r in a.runs:
            fr = frames(r, eid)
            p = a.out / f"{r.name}__{eid.replace(':', '_')}.mp4"
            imageio.mimwrite(p, fr, fps=FPS, macro_block_size=1)
            print(f"-> {p} ({len(fr)} frames)")
            clips.append(fr)
        n = max(len(c) for c in clips)
        side = [np.concatenate([c[min(i, len(c) - 1)] for c in clips], axis=1) for i in range(n)]
        p = a.out / f"side_by_side__{eid.replace(':', '_')}.mp4"
        imageio.mimwrite(p, side, fps=FPS, macro_block_size=1)
        print(f"-> {p} ({n} frames)")


if __name__ == "__main__":
    main()
