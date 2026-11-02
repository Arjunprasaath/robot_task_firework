"""Finish (or build) an SFT set by reading lerobot/metaworld_mt50 parquet files directly.

Fallback for when the HF datasets-server rows API is down (it returned 503 on 2026-11-02).
Same episodes in the same order (first N successful per task by episode_index), same prompt,
image pipeline and answer format as build_sft.py. Episodes already in <out>/index.csv are kept
and skipped, new rows are appended. Each ~100 MB parquet file (one row group, a few episodes)
is downloaded once, used, and deleted.

    uv run --locked python -m ml45bench.build_sft_parquet --split train45 --episodes 20 --stride 4 --out data/sft_train45
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import io
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from . import spec
from .build_sft import HF, get

COLS = ["observation.state", "action", "next.success", "observation.environment_state",
        "observation.image", "frame_index", "episode_index"]


def png_to_jpeg_url(png: bytes) -> str:
    img = Image.open(io.BytesIO(png)).convert("RGB").resize((spec.IMAGE_SIZE, spec.IMAGE_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=spec.JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Files:
    """Download each parquet file once (shared between tasks), drop it when no longer needed."""

    def __init__(self, cache: Path):
        self.cache, self.lock, self.locks, self.tables = cache, threading.Lock(), {}, {}
        cache.mkdir(parents=True, exist_ok=True)

    def table(self, c: int, f: int):
        key = (c, f)
        with self.lock:
            lk = self.locks.setdefault(key, threading.Lock())
        with lk:
            if key not in self.tables:
                p = self.cache / f"c{c:03d}_f{f:03d}.parquet"
                t0 = time.time()
                p.write_bytes(get(f"{HF}/data/chunk-{c:03d}/file-{f:03d}.parquet"))
                self.tables[key] = pq.read_table(p, columns=COLS)
                p.unlink()
                print(f"  file {c}/{f}: {self.tables[key].num_rows} rows in {time.time() - t0:.0f}s", flush=True)
            return self.tables[key]

    def drop(self, c: int, f: int):
        with self.lock:
            self.tables.pop((c, f), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train45", "test5"], required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = spec.TRAIN_TASK_INDICES if args.split == "train45" else list(spec.TEST_TASKS)
    by_text = {spec.TASK_TEXT[t]: t for t in tasks}

    have = collections.defaultdict(set)
    idx_path = out / "index.csv"
    if idx_path.exists():
        for r in csv.DictReader(open(idx_path)):
            have[int(r["task_index"])].add(int(r["episode_index"]))
    new_idx = not idx_path.exists()

    meta = pq.read_table(io.BytesIO(get(f"{HF}/meta/episodes/chunk-000/file-000.parquet")),
                         columns=["episode_index", "tasks", "data/chunk_index", "data/file_index"]).to_pylist()
    meta = sorted((e for e in meta if e["tasks"] and e["tasks"][0] in by_text), key=lambda e: int(e["episode_index"]))
    per_task = {t: [e for e in meta if by_text[e["tasks"][0]] == t] for t in tasks}
    todo = {t: args.episodes - len(have[t]) for t in tasks if len(have[t]) < args.episodes}
    print("still needed per task:", todo, flush=True)

    files = Files(out.parent / "_parquet_cache")
    lock = threading.Lock()
    fh = open(out / "train.jsonl", "a")
    ixf = open(idx_path, "a", newline="")
    idx = csv.writer(ixf)
    if new_idx:
        idx.writerow(["task_index", "episode_index", "frame_index", "obj0", "goal0"])
    report = {}

    def do_task(t: int) -> None:
        kept = len(have[t])
        skipped = []
        for e in per_task[t]:
            if kept >= args.episodes:
                break
            ei = int(e["episode_index"])
            if ei in have[t]:
                continue
            c, f = int(e["data/chunk_index"]), int(e["data/file_index"])
            tab = files.table(c, f)
            rows = [r for r in tab.to_pylist() if int(r["episode_index"]) == ei]
            rows.sort(key=lambda r: int(r["frame_index"]))
            if not rows or not any(bool(r["next.success"]) for r in rows):
                skipped.append(ei)
                continue
            kept += 1
            T = len(rows)
            env0 = rows[0]["observation.environment_state"]
            scene = (json.dumps([round(v, 4) for v in env0[4:7]]), json.dumps([round(v, 4) for v in env0[-3:]]))
            lines = []
            for i in range(0, T - 1, args.stride):
                r = rows[i]
                st = r["observation.state"]
                ee = [round(float(x), 3) for x in st[:3]]
                closed = int(float(st[3]) < spec.GRIPPER_CLOSED_BELOW)
                acts = [spec.quantize(rows[min(i + k, T - 1)]["action"]) for k in range(spec.CHUNK)]
                answer = ";".join(",".join(map(str, a)) for a in acts)
                assert spec.ANSWER_RE.fullmatch(answer), answer
                url = png_to_jpeg_url(r["observation.image"]["bytes"])
                msgs = spec.messages(t, ee, closed, url) + [{"role": "assistant", "content": answer}]
                lines.append((json.dumps({"messages": msgs}), [t, ei, int(r["frame_index"]), *scene]))
            with lock:
                for j, ix in lines:
                    fh.write(j + "\n")
                    idx.writerow(ix)
                fh.flush()
                ixf.flush()
            print(f"task {t:2d} ep {ei} rows {len(lines)} ({kept}/{args.episodes}) [parquet]", flush=True)
        report[t] = {"kept_total": kept, "skipped_failed_parquet": skipped}

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(do_task, list(todo)))
    fh.close()
    ixf.close()
    counts = collections.Counter(int(r["task_index"]) for r in csv.DictReader(open(idx_path)))
    eps = collections.defaultdict(set)
    for r in csv.DictReader(open(idx_path)):
        eps[int(r["task_index"])].add(int(r["episode_index"]))
    n = sum(1 for _ in open(out / "train.jsonl"))
    (out / "stats.json").write_text(json.dumps({
        "split": args.split, "episodes_per_task": args.episodes, "stride": args.stride, "rows": n,
        "spec_fingerprint": spec.spec_fingerprint(),
        "per_task": {t: {"episodes": sorted(eps[t]), "rows": counts[t]} for t in tasks},
        "parquet_fill": report,
        "sources": "rows from HF datasets-server (JPEG-served images) until it returned 503, rest from parquet (PNG)"},
        indent=1))
    short = {t: len(eps[t]) for t in tasks if len(eps[t]) < args.episodes}
    print(f"done: {n} rows -> {out}; tasks short of {args.episodes}: {short or 'none'}")


if __name__ == "__main__":
    main()
