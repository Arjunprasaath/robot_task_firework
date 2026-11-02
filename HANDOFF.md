# HANDOFF: ML45 VLM robot-policy benchmark on Fireworks

Written 2026-11-02, 11:55 PT, for the next agent working with Arjun (repo owner) and their teammate.
Read all of it before acting. Section 0 has the rules the user set; breaking them has cost trust before.

---

## 0. Working rules from the user (follow strictly)

1. **Ask before making assumptions.** The user said so explicitly ("why are you making assumptions, ask me before making"). If a design choice is not written here, ask.
2. **Fireworks only, for every model call and all training.** Do **not** use the SSH server "Anton" (DGX Spark) any more. The simulator runs locally on the user's Mac (or any Linux box).
3. **API key:** in `.env` as `FIREWORK_API=...`. **Never print, cat, echo or commit it.** Code reads it by name (`ml45bench/fireworks.py: api_key()`). `.env` is git-ignored.
4. **Don't overload the Mac.** A launch with 40 Python processes and in-memory video froze the user's machine. Use the single-process runner (section 4.3) and one model at a time unless asked otherwise.
5. **Confirm before spending real money.** The user said "Wait" for the big deployments (Gemma 4 on 4×B200 ≈ \$52/hr; Nemotron on 4×B300 ≈ \$60/hr). Their latest instruction: **first check how Qwen does, then decide.** Qwen (1×H100) and Glimmer (1×H200) are ≈ \$8/hr each; using them for the eval is approved.
6. **Delete deployments when done.** Nothing is deployed right now (checked 11:53).
7. **GitHub repo is public:** https://github.com/Arjunprasaath/robot_task_firework. The user had the old code and plans wiped from it ("why have you uploaded the old plans and code?"). Push only the benchmark work, never secrets, logs, data or runs.
8. **Disclose AI-assistant use** in the submission (event rule).
9. Keep docs concise. The user found earlier docs too detailed. This handoff is the exception (they asked for detail).

---

## 1. Goal and deadline

**Event:** Fireworks "Deploy" Buildathon (with Microsoft and NVIDIA), SF, **today, Nov 2 2026**.
- Building ends 4:30 PM PT. Submissions open at 3:30. **Demo video (< 120 s) due 4:00.**
- Submit a public GitHub repo plus a commit hash.
- Rules that matter:
  - Beat a "frontier" baseline on the same eval cases. A judge confirmed that **any open model in the Fireworks catalog counts as frontier**.
  - Eval cases must be separate from the training data, with **100+ varied held-out examples** and **uncertainty estimates**.
  - No credentials in the repo. Disclose prior work and AI use. Prizes require using Fireworks.
- Judging: 40% benchmark, 25% business value, 20% technical, 10% presentation, 5% documentation.

**Project (user's words):**
1. Train **NVIDIA Nemotron 3.5 Super VL 120B A12B BF16** on the **ML45 training split (45 MetaWorld tasks)**.
2. Evaluate it on the **5 held-out ML45 test tasks**.
3. Do a **5-shot** fine-tune (5 demos per test task) and **re-evaluate**.
4. Compare against base models on the same benchmark: **Muse Glimmer 30B (Meta), Nemotron 3.5 Super VL (base), Qwen3.8 27B, Gemma 4 26B A4B IT**.
5. The eval must be **constant and replicable** across models.

Choices the user has made:
- Strict ML45 (zero-shot on held-out tasks, then 5-shot).
- ~20 episodes per task at stride 4.
- No RL.
- Goal names used as written.
- LoRA rank above 16 (64 proposed).
- One epoch first, with more epochs later via warm-start if time allows.

---

## 2. Where everything is

| What | Path / link |
|---|---|
| Agent worktree (where all of this was built) | `/Users/arjunprasaath/.local/share/openresearch/worktrees/cabc8df6-61c2-4cf8-9081-1a0c61658851/chat_7c1ca14e-1c04-428a-b3c2-4e1634c3087c` (branch `ml45-bench`, pushed as GitHub `main`; git remote name `github`) |
| User's own clone | `/Users/arjunprasaath/Projects/Firework hackathon/robot_task_firework` (holds the real `.env`; **still on the old, now-deleted history**: needs `git fetch && git reset --hard origin/main`, ask first) |
| `.env` in the worktree | symlink to the user's clone's `.env` |
| GitHub | https://github.com/Arjunprasaath/robot_task_firework (public) |
| Fireworks account | `arjunprasaath-us-fgm` |
| Python env | `uv sync --locked` (Python 3.11, `pyproject.toml` + `uv.lock`). Run everything as `uv run --locked python -m ml45bench.<module>` |
| Old work (reference only, not in the repo) | Branches `orx/retrain-with-fixed-label-mask` etc. in the same local git repo; Anton `~/Projects/firework_ai_hackathon`; artifacts in `/Users/arjunprasaath/.local/share/openresearch/files/robotic-finetuning/` |

---

## 3. Current status (11:55 PT)

### Done and verified
- **Benchmark frozen** (`benchmark/manifest.json`, `spec_fingerprint` in file):
  - 100 held-out episodes: 5 tasks × seeds 1000–1019.
  - `manifest --check` rebuilds in a fresh process: **MATCH**.
- **Ceiling:** MetaWorld's scripted expert solves **100/100**, raw and through our quantized 3-number action codec.
  - Steps to success: bin-picking 92–138, box-close 90–141, door-lock 58–119, door-unlock 37–52, hand-insert 53–65.
- **Runner** (new single-process version, uncommitted until this handoff's commit):
  - Expert run at concurrency 20: 100/100 in 115 s, about 1 GB RAM.
  - `--replay` → **REPLAY MATCH** on all 100.
- **Fireworks smoke tests** (one real benchmark request each, before deployments were deleted):
  - **Qwen3.8 27B:** valid 3-action reply, 314 prompt + 25 completion tokens, 2.8 s, 0 reasoning tokens.
  - **Muse Glimmer 30B:** rejects `reasoning_effort: "none"` (HTTP 400, "thinking-only model"). With `"low"` and the grammar it returns a valid answer, 24 completion tokens, 0 reasoning chars, ~3.5 s. `models.json` now uses `"low"` for Glimmer (documented there).
- An earlier full Qwen/Glimmer run was **killed** (it froze the Mac). Its partial data was deleted. **No model results exist yet.**

### In progress
- **ML45 training-set build** (background PID 17817/17819, started 11:41):
  - Command: `uv run --locked python -m ml45bench.build_sft --split train45 --episodes 20 --stride 4 --workers 8 --out data/sft_train45`. Log: `logs/build_train45.log`.
  - At 11:53: 3,390 rows (56 MB). Expected total ≈ 45 × 20 episodes × ~20 rows ≈ **18k rows ≈ 300 MB**.
  - It slowed down (HF datasets-server rate limiting, handled with backoff). **ETA possibly ~12:45–1:00.**
  - Check with `wc -l data/sft_train45/train.jsonl; tail logs/build_train45.log`. When finished it prints `done: N rows` and writes `stats.json`.
  - Rows are appended per episode as each completes, so the file is always valid JSONL. A row is
    `{"messages":[system, user(text+image_url data:jpeg base64), assistant("dx,dy,dz,g;…;…")]}`.

### Not started
- Upload the dataset to Fireworks plus the SFT job (section 6).
- 5-shot dataset and job.
- Any model eval run (Qwen first, per the user).
- Final report, chart, demo video, README results section.

---

## 4. The benchmark (code tour)

```
ml45bench/
  spec.py        THE frozen definition: split, task texts, goal names, seeds, prompt, grammar, decoding, codec
  sim.py         MetaWorld env + per-episode mujoco.Renderer, observation (JPEG), proprio, Episode.step
  manifest.py    builds/checks benchmark/manifest.json; scripted-expert ceiling
  run.py         runs a model on the 100 episodes (single process + HTTP thread pool); --replay
  report.py      Wilson CIs, per-task, progress, tokens, latency; paired McNemar + bootstrap across runs
  fireworks.py   REST client: chat (retries 429/5xx, never alters request), deployments CRUD
  deploy.py      up / status / probe / down for models in benchmark/models.json
  build_sft.py   builds Fireworks SFT JSONL from lerobot/metaworld_mt50 with the exact benchmark prompt
benchmark/
  manifest.json  100 episodes: id, task, seed, initial obj/goal pos, ee0, sha of first image, expert results
  models.json    per-model Fireworks base model, deployment id, pinned deployment shape, extra request params
```

### 4.1 Protocol (all in `spec.py`; changing anything there breaks the fingerprint, so bump `BENCH_VERSION` and rebuild the manifest)
- **Split (ML45, Yu et al. 2019, arXiv:1910.10897):**
  - Test tasks, as MT50 `task_index` → env: 2 `bin-picking-v3`, 3 `box-close-v3`, 14 `door-lock-v3`, 16 `door-unlock-v3`, 17 `hand-insert-v3`.
  - Training = the other 45 MT50 indices.
- **Episodes:**
  - The env is built with `gym.make("Meta-World/MT1", env_name, seed=seed)` and `reset(seed=seed)`, with a fresh env per episode. MT1 fixes positions at construction, so `reset(seed)` alone does not move objects.
  - At most 500 steps. Success = `info["success"]` at any step.
- **Observation:**
  - `corner2` camera with `cam_pos = (0.75, 0.075, 0.7)` (the LeRobot MT50 dataset pose; MetaWorld's default is (1.3, −0.2, 1.1)), then `mj_forward`.
  - Rendered 480×480 with `mujoco.Renderer`, rotated 180° (`np.flip(.., (0,1))`), resized bicubic to 256, JPEG q85, base64 data URL.
  - Same pipeline as the training images (dataset 480 px → 256 bicubic → q85).
- **Proprio:** `obs[:3]` rounded to mm. `Gripper: closed` iff `obs[3] < 0.5`. Dataset `observation.state == obs[:4]` (verified).
- **Prompt:**
  - System: `system_prompt()` (3 actions, format explained, example).
  - User: `Task: <exact MT50 text>\nGoal: <GOAL_NAMES[t]>\nEnd effector: x=…, y=…, z=…\nGripper: open|closed\nPredict the next 3 actions.` followed by the image.
  - The goal is named, never located (team decision).
- **Action:**
  - `dx,dy,dz,grip` ×3, `;`-separated. xyz ∈ −10..10 (10 = 1 cm = MetaWorld action 1.0); grip ∈ −2..2 → /2.
  - The 3 actions are executed open-loop, then the model is called again.
- **Decoding (identical for all models):**
  - `temperature 0, top_p 1, max_tokens 64, seed 0`, `response_format={"type":"grammar","grammar":GBNF}` for exactly 3 actions.
  - `reasoning_effort` per model in `models.json`: `"none"` everywhere except Glimmer (`"low"`).
- **Invalid reply:** execute the valid actions found (at most 3), pad with no-ops, count in `invalid_reply_rate`.
- **Metrics:**
  - Success with 95% Wilson CI, overall and per task.
  - Partial progress = max dense reward / 10; grasp rate; `min_obj_to_target`.
  - Calls; invalid rate; prompt and completion tokens; per-call latency p50/p95; model time per episode; time to success.
  - Pairwise across runs: exact McNemar and paired-bootstrap 95% CI of the success difference (`report.py`).

### 4.2 Replicability guarantees
- Before each episode the runner checks the initial scene and the sha256 of the first image against the manifest, so drift in renderer or library versions is detected.
- Every call is logged in `runs/<run>/calls/<episode>.jsonl`: image sha, user text, raw reply, parsed actions, latency, tokens, finish reason, reasoning chars. Each episode's result is in `runs/<run>/episodes/*.json`; the run config in `run.json` (address, request template, spec fingerprint, manifest sha, git commit).
- `python -m ml45bench.run --replay runs/<run>` re-simulates from the logs without the API; it must print `REPLAY MATCH`.
- Temperature 0 on an API is not guaranteed bit-identical across calls. The log is the record, not a re-query. Say this in the README and the presentation.
- Runs resume: re-running the same `--model` skips finished episodes, and refuses if the config changed (then use `--run-name`).

### 4.3 Runner design (why it is this way)
- **One process.** The main thread owns all simulators and renderers. A `ThreadPoolExecutor(concurrency)` only waits on HTTP.
  - Sim ≈ 10 ms per call; a model reply ≈ 2–3 s. So speed = concurrency, and the Mac stays cool.
- **Renderer:**
  - Use one `mujoco.Renderer` per episode, closed at episode end.
  - **Do not go back to gymnasium's `render_mode="rgb_array"`.** Its glfw viewer gave *stale frames* when many envs lived in one process: after the first episode closed, later episodes' images stopped changing. Replay caught it.
  - The mujoco.Renderer images are byte-identical to the old ones (`manifest --check` MATCH).
- `--video` streams frames to `runs/<run>/videos/*.mp4` (no RAM build-up). It adds disk I/O; leave it off for scoring runs unless needed for the demo.
- Old bug to avoid: 2 runs × 20 *processes* with in-memory frames froze the Mac.

---

## 5. Fireworks facts (verified this session via API / docs)

### 5.1 Models (all `supportsImageInput: true`, `supportsServerless: false`, LoRA SFT supported)

| key | model id | shape pinned in models.json | GPUs | ≈ \$/hr | notes |
|---|---|---|---|---|---|
| `qwen3p8-27b` | `accounts/fireworks/models/qwen3p8-27b` | `deploymentShapes/sva2n3mh` | 1× H100 | 8 | works, reasoning `none` OK |
| `glimmer-30b` | `accounts/fireworks/models/muse-glimmer-30b` ("Meta Glimmer" = Muse Glimmer 30B, HF `meta-models/Muse-Glimmer-30B`) | `deploymentShapes/dkuwotf7` | 1× H200 | 8 | thinking-only: use `reasoning_effort: "low"` |
| `gemma4-26b-a4b` | `accounts/fireworks/models/gemma-4-26b-a4b-it` | `deploymentShapes/rft-gemma-4-26b-a4b-it` | 4× B200 | ~52 | the non-RFT shape `gemma4-26b-a4b-it` (4×H200) is **blocked** by Fireworks (a "TGV BF16 GEMM regression" message); shapeless 1×H100/H200 is "not available" |
| `nemotron-super-vl` | `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` | `deploymentShapes/rft-nemotron-3-5-super-vl-ea1008-bf16-262k-b300-tp-ep` | 4× B300 | ~60 | the only shape; `default` resolves to it; **no MULTI_LORA**, so `enableAddons` is rejected |
| `nemotron-ml45-sft` / `-5shot` | `accounts/{account}/models/nemotron-ml45-sft` / `…-5shot` (names to create) | same Nemotron shape | 4× B300 | ~60 each | serve each fine-tuned LoRA **live-merged** as its own deployment (deploy the fine-tuned model as `baseModel`) |

Also available: `nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4` (vision, FP4, 4×B200 default shape; no LoRA). Serverless vision models exist (e.g. `qwen3p8-max`, `kimi-k3`, `glm-5p3-flash`) if a cheap extra baseline is wanted. Ask first.

Pricing (fireworks.ai/pricing): H100/H200 \$8/hr, B200 \$13/hr, B300 \$15/hr per GPU, billed per GPU-second. A 4-GPU shape is 4× that.

### 5.2 API details
- **Chat:** `POST https://api.fireworks.ai/inference/v1/chat/completions`, `model: "<model>#accounts/<acct>/deployments/<id>"` (built by `run.model_address`).
- **Grammar:** `response_format: {"type":"grammar","grammar":"<GBNF>"}` works together with images (Qwen, Glimmer tested).
- **Reasoning:** `reasoning_effort: "none"` disables it (Qwen 3.8 OK); thinking-only models reject `"none"`. `seed` is accepted. Images go as `image_url` data URLs (base64 total < 10 MB per request).
- **Deployments** (`fireworks.py`):
  - Create: `POST /v1/accounts/{acct}/deployments?deploymentId=…`, body `{baseModel, deploymentShape, minReplicaCount:0, maxReplicaCount:1, autoscalingPolicy:{scaleToZeroWindow:"900s"}}`. Shapes are required in practice.
  - Validate without creating: add `validateOnly=true`.
  - List shapes: `GET /v1/accounts/fireworks/deploymentShapes/-/versions?filter=snapshot.base_model="accounts/fireworks/models/<id>"`.
  - Delete: `DELETE …/deployments/{id}?ignoreChecks=true`.
  - Cold start ≈ 10–15 min. A scaled-to-zero deployment returns 503 while scaling up; the client retries.
  - **`deploymentShape: "default"` picked 4×B200 for Gemma.** Always pin the shape.
- **SFT (docs, not yet exercised):**
  - `POST /v1/accounts/{acct}/supervisedFineTuningJobs?supervisedFineTuningJobId=…`
  - Body fields: `dataset` (required, `accounts/{acct}/datasets/<id>`), `baseModel` **or** `warmStartFrom` (a previous LoRA model, for continued training / 5-shot), `outputModel`, `epochs`, `learningRate`, `loraRank`, `evaluationDataset`, `maxContextLength`, `batchSizeSamples`, `learningRateWarmupSteps`, `lrScheduler`.
  - Unresolved: the docs show both a flat body and one wrapped in `"supervisedFineTuningJob": {...}`. Try flat first; if fields are ignored, wrap.
  - Max `loraRank` for this model is unknown; try 64.
  - `firectl` is **not installed** locally; use REST or install `firectl` (ask).
- **Datasets:**
  - Create: `POST /v1/accounts/{acct}/datasets` body `{"datasetId": "...", "dataset": {"displayName": "...", "format": "CHAT"}}`.
  - Get signed URLs: `POST …/datasets/{id}:getUploadEndpoint` body `{"filenameToSize": {"train.jsonl": "<bytes as string>"}}`. Then upload to the returned URL; the method is probably `PUT` (unverified).
  - Validate: `POST …/datasets/{id}:validateUpload` body `{}`.
  - Single-request upload is limited to 150 MB, and our file will be ~300 MB, so use the signed URL.

---

## 6. Next steps (suggested order; confirm money-spending steps with the user)

1. **Commit state.** Done in this handoff's commit: runner fix, `build_sft.py`, this file.
2. **Qwen eval** (approved, ~\$8/hr):
   ```bash
   uv run --locked python -m ml45bench.deploy up qwen3p8-27b      # wait for READY (~10-15 min): deploy status
   uv run --locked python -m ml45bench.deploy probe qwen3p8-27b
   uv run --locked python -m ml45bench.run --model qwen3p8-27b --concurrency 20   # ~25-30 min expected
   uv run --locked python -m ml45bench.run --replay runs/qwen3p8-27b              # must say REPLAY MATCH
   uv run --locked python -m ml45bench.report runs/qwen3p8-27b
   uv run --locked python -m ml45bench.deploy down ml45-qwen3p8-27b
   ```
   Then show the user the numbers and ask about Glimmer, Gemma and Nemotron base (costs in 5.1).
   Expectation: zero-shot success near 0 on held-out tasks. Partial progress and grasp rate are the informative columns.
3. **When the dataset build finishes:**
   - Check `stats.json` for kept/skipped episodes per task.
   - Do one sanity check: random rows decode, `ANSWER_RE` matches, the image opens.
   - Create the Fireworks dataset and upload `data/sft_train45/train.jsonl` via the signed URL.
   - Optionally hold out a small eval set (e.g. the last episode of each task) for `evaluationDataset`.
   - **Do not upload the 5 test tasks' data in this job.**
4. **SFT job** (confirm cost with the user first; the price was not checked this session):
   - `baseModel = nvidia-nemotron-3p5-super-vl-120b-a12b-bf16`, `loraRank 64`, `epochs 1`, default LR, `outputModel nemotron-ml45-sft`.
   - Poll the job state.
   - The user wants the option of more epochs later: a new job with `warmStartFrom = accounts/{acct}/models/nemotron-ml45-sft`.
5. **Deploy and eval the fine-tuned model:**
   - Set `models.json` `nemotron-ml45-sft.serve_model` to the real output model name, then `deploy up nemotron-ml45-sft`, `probe`, `run`, `replay`, `down`.
   - Run the **Nemotron base** eval as well (needs user approval) so the fine-tuning gain is a paired comparison.
6. **5-shot:**
   - Build: `uv run --locked python -m ml45bench.build_sft --split test5 --episodes 5 --stride 2 --out data/sft_5shot`. This takes the first 5 successful MT50 demos per test task.
   - **Before training, check that no demo's initial scene (index.csv `obj0`/`goal0`) is close to any manifest eval scene** (e.g. obj and goal within 1 cm). Report the minimum distance. MT50 demo scenes come from a different RNG stream than seeds 1000–1019, but verify.
   - Train with `warmStartFrom = nemotron-ml45-sft`, few epochs (small data), output `nemotron-ml45-5shot`, then deploy and eval as in step 5.
7. **Report:**
   - `python -m ml45bench.report runs/<a> runs/<b> …` → `runs/comparison.json`.
   - Add a results table to the README: success with CI, per task, progress, tokens per episode, latency, \$ per run.
   - Make a success-vs-cost chart.
   - Record a short video with `--video` on a few episodes (side-by-side base vs fine-tuned).
   - Commit, push, and give the user the commit hash.

**Time risk:**
- SFT duration for a 120B VL model on Fireworks is unknown.
- If 45-task SFT is not done by ~2:45, drop 5-shot and present: base models vs (SFT if ready), with the benchmark, replay and cost analysis as the technical contribution.
- Tell the user early.

---

## 7. Open questions for the user
- Approve the Glimmer eval (\$8/hr)? Gemma (4×B200) and Nemotron base (4×B300)? After seeing Qwen.
- SFT budget, epochs, and whether to carve out an `evaluationDataset`.
- `max_tokens 64` and 500-step budget: fine as frozen? Changing either means a new benchmark version.
- Should the user's local clone be reset to the new GitHub history?

---

## 8. Background: what happened before today (for context, not in the repo)
- **Before:** Nemotron **Nano 12B VL** QLoRA, trained locally on Anton, on MT50 `assembly` only (5-action chunks).
  - Final: 1/5 closed-loop success (seed 2: 120 steps, 146.7 s, 13,248 tokens). The base model failed (21,965 tokens).
  - Lessons, all carried into this benchmark:
    - The label mask was off by `<think></think>` (5 tokens), so answer starts were never trained.
    - The "copy action 1" shortcut, which led to using 3 actions per call.
    - Eval loss looked fine while generations collapsed, so generate before trusting loss.
    - The tokenizer splits `,-`, so the grammar must match the training answers.
    - Camera pose mismatch (pixel MSE 5010 → 37 after fixing to (0.75, 0.075, 0.7) + `mj_forward`).
    - Gripper threshold mismatch (0.05 vs 0.5).
    - `reset(seed)` doesn't move objects, so build a new env per episode.
    - The policy didn't aim at the goal, since the goal location isn't in the prompt.
- **HF datasets** (private, org `hack-fireworks-ai`): `metaworld-mt50-assembly-sft` and `metaworld-mt50-5task-sft`.
  - **Do not use the 5-task set for ML45.** It contains box-close, a held-out test task, and uses an older prompt.
- **Camera check per test task** (dataset frame vs sim frame) was only done for assembly. The camera pose is global, so it should hold. A quick visual check on one test task is cheap if time permits.

## 9. Gotchas
- The `logs/`, `data/`, `runs/` and `.env` directories are git-ignored. Keep it that way.
- `uv run --locked`: if dependencies change, run `uv lock` and commit `uv.lock`.
- `fireworks.chat` retries 429/5xx/timeouts up to 12 times with backoff and never changes the request. Other 4xx raise `APIError` and stop the run. That is intended: fix the config, don't paper over it.
- `models.json` `extra` is merged *after* `spec.DECODING`, so it can override decoding. Only use it for model-required switches (reasoning) and document why.
- Glimmer under the grammar returns no reasoning, but log `reasoning_chars` anyway.
- macOS renders through `MUJOCO_GL=glfw` (set in `sim.py`); `cgl` crashes in gymnasium. Linux uses `egl`.
