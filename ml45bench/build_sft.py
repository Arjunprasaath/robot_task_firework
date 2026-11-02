"""Build Fireworks SFT JSONL from lerobot/metaworld_mt50 with the benchmark's exact prompt.

    # ML45 training set: the 45 train tasks, first 20 successful demos each, every 4th frame
    uv run --locked python -m ml45bench.build_sft --split train45 --episodes 20 --stride 4 --out data/sft_train45
    # 5-shot set: the 5 held-out tasks, first 5 successful demos each
    uv run --locked python -m ml45bench.build_sft --split test5 --episodes 5 --stride 2 --out data/sft_5shot

Rows come from the HF datasets-server (the full repo is ~94 GB of PNG-in-parquet).
Only demos with next.success are used. The answer for frame i is the next CHUNK dataset actions,
quantized exactly like the eval decodes them. Writes train.jsonl ({"messages": [...]} per line),
index.csv (row -> task, episode, frame, initial scene) and stats.json.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from . import spec

HF = "https://huggingface.co/datasets/lerobot/metaworld_mt50/resolve/main"
ROWS = "https://datasets-server.huggingface.co/rows"


def get(url: str, retries: int = 12) -> bytes:
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 429 and e.code < 500:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(ra) if ra and ra.isdigit() else min(120, 2 * 2 ** a) * (0.5 + random.random()))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 2 ** a))
    raise RuntimeError(f"GET failed {url}: {last}")


def episodes_meta() -> list[dict]:
    return pq.read_table(io.BytesIO(get(f"{HF}/meta/episodes/chunk-000/file-000.parquet")),
                         columns=["episode_index", "dataset_from_index", "dataset_to_index", "tasks", "length"]).to_pylist()


def fetch_rows(start: int, end: int) -> list[dict]:
    rows, off = [], start
    while off < end:
        q = urllib.parse.urlencode({"dataset": "lerobot/metaworld_mt50", "config": "default", "split": "train",
                                    "offset": off, "length": min(100, end - off)})
        chunk = json.loads(get(f"{ROWS}?{q}"))["rows"]
        if not chunk:
            break
        rows += [c["row"] for c in chunk]
        off += len(chunk)
    return rows


def to_jpeg_url(src: str) -> str:
    img = Image.open(io.BytesIO(get(src))).convert("RGB").resize((spec.IMAGE_SIZE, spec.IMAGE_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=spec.JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ok(rows: list[dict]) -> bool:
    return any(bool(any(v) if isinstance(v, list) else v) for v in (r.get("next.success") for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train45", "test5"], required=True)
    ap.add_argument("--episodes", type=int, default=20, help="successful demos per task")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = spec.TRAIN_TASK_INDICES if args.split == "train45" else list(spec.TEST_TASKS)
    by_text = {spec.TASK_TEXT[t]: t for t in tasks}
    meta = sorted((e for e in episodes_meta() if e["tasks"] and e["tasks"][0] in by_text),
                  key=lambda e: int(e["episode_index"]))
    per_task = {t: [e for e in meta if by_text[e["tasks"][0]] == t] for t in tasks}

    lock = threading.Lock()
    fh = open(out / "train.jsonl", "w")
    idx = csv.writer(open(out / "index.csv", "w", newline=""))
    idx.writerow(["task_index", "episode_index", "frame_index", "obj0", "goal0"])
    stats = {t: {"kept": [], "skipped_failed": []} for t in tasks}
    pool = ThreadPoolExecutor(args.workers)

    def do_task(t: int) -> None:
        kept = 0
        for e in per_task[t]:
            if kept >= args.episodes:
                break
            rows = fetch_rows(int(e["dataset_from_index"]), int(e["dataset_to_index"]))
            if not rows or not ok(rows):
                stats[t]["skipped_failed"].append(int(e["episode_index"]))
                continue
            kept += 1
            T = len(rows)
            frames = list(range(0, T - 1, args.stride))
            env0 = rows[0]["observation.environment_state"]
            scene = (json.dumps([round(v, 4) for v in env0[4:7]]), json.dumps([round(v, 4) for v in env0[-3:]]))
            lines = []
            for i in frames:
                r = rows[i]
                st = r["observation.state"]
                ee = [round(float(x), 3) for x in st[:3]]
                closed = int(float(st[3]) < spec.GRIPPER_CLOSED_BELOW)
                acts = [spec.quantize(rows[min(i + k, T - 1)]["action"]) for k in range(spec.CHUNK)]
                answer = ";".join(",".join(map(str, a)) for a in acts)
                assert spec.ANSWER_RE.fullmatch(answer), answer
                url = to_jpeg_url(r["observation.image"]["src"])
                msgs = spec.messages(t, ee, closed, url) + [{"role": "assistant", "content": answer}]
                lines.append((json.dumps({"messages": msgs}), [t, int(e["episode_index"]), int(r["frame_index"]), *scene]))
            with lock:
                for j, ix in lines:
                    fh.write(j + "\n")
                    idx.writerow(ix)
                fh.flush()
                stats[t]["kept"].append(int(e["episode_index"]))
            print(f"task {t:2d} ep {e['episode_index']} rows {len(lines)} ({kept}/{args.episodes})", flush=True)

    list(pool.map(do_task, tasks))
    fh.close()
    n = sum(1 for _ in open(out / "train.jsonl"))
    (out / "stats.json").write_text(json.dumps({
        "split": args.split, "episodes_per_task": args.episodes, "stride": args.stride, "rows": n,
        "spec_fingerprint": spec.spec_fingerprint(), "tasks": stats}, indent=1))
    print(f"done: {n} rows -> {out}")


if __name__ == "__main__":
    main()
