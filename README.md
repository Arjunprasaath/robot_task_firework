# robot_task_firework: an ML45 held-out benchmark for VLM robot policies

Can a vision-language model drive a robot arm on manipulation tasks it was never trained on?
Models control a simulated Sawyer arm in [Meta-World](https://arxiv.org/abs/1910.10897) from one
camera image plus the end-effector state, and reply with low-level actions. Training and inference
run on Fireworks; the simulator runs locally.

## Protocol (frozen in `ml45bench/spec.py` and `benchmark/manifest.json`)

| | |
|---|---|
| Split | Meta-World **ML45**: train on 45 tasks, test on 5 held-out tasks: `bin-picking`, `box-close`, `door-lock`, `door-unlock`, `hand-insert` |
| Test set | 5 tasks × 20 seeds (1000–1019) = **100 episodes**, each a different scene (object and goal positions logged in the manifest) |
| Episode | ≤ 500 env steps (MetaWorld `max_path_length`); success = `info["success"]` at any step |
| Observation | `corner2` camera at the LeRobot MT50 pose, 480 px → 256 px bicubic → JPEG q85, plus `End effector: x,y,z` and `Gripper: open/closed` (`obs[3] < 0.5`) |
| Prompt | Task text from `lerobot/metaworld_mt50` plus `Goal: <name>`. The goal is named but its location is never given |
| Action | 3 actions per call, each `dx,dy,dz,grip`; xyz ∈ −10..10 (10 = 1 cm), grip ∈ −2..2, executed open-loop |
| Decoding | same for every model: temperature 0, top_p 1, seed 0, max_tokens 64, GBNF grammar for exactly 3 actions, reasoning off |
| Invalid reply | the valid actions in it are executed, padded with no-ops; counted in `invalid_reply_rate` |
| Reference | MetaWorld's scripted expert solves 100/100 episodes, also through our quantized action format (ceiling) |

Replicability:
- The manifest freezes each episode's initial scene and the sha256 of the first image. The runner checks both before every episode, so a different MuJoCo, MetaWorld or Pillow build that would change the episodes is caught.
- `manifest --check` rebuilds the manifest and compares.
- Every model call is logged: image hash, prompt, raw reply, parsed actions, latency and tokens.
- `run --replay` re-simulates a run from its log without the API and checks every image and outcome.
- Dependencies are pinned in `uv.lock`.
- API outputs at temperature 0 are not guaranteed bit-identical across calls, which is why the log, not a re-query, is the record.

## Models (`benchmark/models.json`)

| key | model | Fireworks deployment shape |
|---|---|---|
| `glimmer-30b` | Muse Glimmer 30B (Meta) | 1× H200 |
| `qwen3p8-27b` | Qwen3.8 27B | 1× H100 |
| `gemma4-26b-a4b` | Gemma 4 26B A4B IT | 4× B200 |
| `nemotron-super-vl` | NVIDIA Nemotron 3.5 Super VL 120B A12B BF16 | 4× B300 |
| `nemotron-ml45-sft` | Nemotron + LoRA SFT on the 45 training tasks (zero-shot on test) | 4× B300, live-merged |
| `nemotron-ml45-5shot` | the above + LoRA warm-start on 5 demos per test task | 4× B300, live-merged |

## Results (zero-shot base models, 100 held-out episodes)

| model | success (95% Wilson CI) | progress (95% CI) | invalid replies | tokens / episode | latency / call p50, p95 |
|---|---|---|---|---|---|
| Muse Glimmer 30B (Meta) | 0/100 (0–3.7%) | 0.104 (0.085–0.124) | 0% | 62k | 0.89 s, 1.71 s |
| Nemotron 3.5 Super VL 120B A12B (base) | 0/100 (0–3.7%) | 0.097 (0.078–0.117) | 0% | 89k | 0.82 s, 1.83 s |

![Partial progress per held-out task](docs/figures/progress_by_task.png)

![Tokens and latency per model](docs/figures/efficiency.png)

Figures: `uv run --locked --with matplotlib python -m ml45bench.plots runs/glimmer-30b runs/nemotron-super-vl`.

- Both runs replay exactly from their logs (`REPLAY MATCH`).
- Neither model solves a held-out task zero-shot. The paired comparison has no discordant episodes.
- The fine-tuned Nemotron runs (ML45 SFT, then 5-shot) are pending: Fireworks dataset uploads were disabled for our account. `ml45bench/sft.py` holds the upload and training commands.

## Usage

```bash
uv sync --locked
echo 'FIREWORK_API=<your key>' > .env                      # git-ignored

uv run --locked python -m ml45bench.manifest --check       # same episodes on this machine?
uv run --locked python -m ml45bench.deploy up qwen3p8-27b  # on-demand deployment (scales to zero after 15 min idle)
uv run --locked python -m ml45bench.deploy probe qwen3p8-27b
uv run --locked python -m ml45bench.run --model qwen3p8-27b --concurrency 10
uv run --locked python -m ml45bench.run --replay runs/qwen3p8-27b
uv run --locked python -m ml45bench.report runs/*/                # Wilson CIs, paired McNemar + bootstrap
uv run --locked python -m ml45bench.deploy down ml45-qwen3p8-27b
```

Metrics per model:
- success rate (95% Wilson CI), overall and per task
- partial progress: max dense reward / 10, and grasp rate
- invalid-reply rate
- tokens per episode and per success
- per-call latency p50/p95, and model time per episode

Models are compared pairwise on the same 100 episodes with an exact McNemar test and a paired-bootstrap CI.
