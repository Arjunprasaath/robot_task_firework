"""The frozen benchmark definition: tasks, split, prompt, action format, grammar, decoding.

Everything a model sees or is scored on is defined here, once, and is shared by the
training-data builder and the evaluator. Changing anything in this file changes the
benchmark: bump BENCH_VERSION and rebuild the manifest (`python -m ml45bench.manifest`).
The manifest stores a hash of this spec (`spec_fingerprint()`), and the runner refuses
to run if they disagree.
"""
from __future__ import annotations

import hashlib
import json
import re

BENCH_VERSION = "ml45-heldout-v1"

# ---------------------------------------------------------------- split (Meta-World ML45)
# MT50 task_index (lerobot/metaworld_mt50 meta/tasks.parquet) -> MetaWorld v3 env name.
# The 5 ML45 meta-test tasks (Yu et al. 2019, arXiv:1910.10897). These are never
# trained on in the zero-shot condition.
TEST_TASKS = {
    2: "bin-picking-v3",
    3: "box-close-v3",
    14: "door-lock-v3",
    16: "door-unlock-v3",
    17: "hand-insert-v3",
}
ALL_TASK_INDICES = list(range(50))
TRAIN_TASK_INDICES = [t for t in ALL_TASK_INDICES if t not in TEST_TASKS]  # the 45

# Exact task strings from lerobot/metaworld_mt50 (the training data uses the same text).
TASK_TEXT = {
    0: "Pick up a nut and place it onto a peg",
    1: "Dunk the basketball into the basket",
    2: "Grasp the puck from one bin and place it into another bin",
    3: "Grasp the cover and close the box with it",
    4: "Press a button from the top",
    5: "Bypass a wall and press a button from the top",
    6: "Press a button",
    7: "Bypass a wall and press a button",
    8: "Push a button on the coffee machine",
    9: "Pull a mug from a coffee machine",
    10: "Push a mug under a coffee machine",
    11: "Rotate a dial 180 degrees",
    12: "Pick a nut out of a peg",
    13: "Close a door with a revolving joint",
    14: "Lock the door by rotating the lock clockwise",
    15: "Open a door with a revolving joint",
    16: "Unlock the door by rotating the lock counter-clockwise",
    17: "Insert the gripper into a hole",
    18: "Push and close a drawer",
    19: "Open a drawer",
    20: "Rotate the faucet counter-clockwise",
    21: "Rotate the faucet clockwise",
    22: "Hammer a screw on the wall",
    23: "Press a handle down sideways",
    24: "Press a handle down",
    25: "Pull a handle up sideways",
    26: "Pull a handle up",
    27: "Pull a lever down 90 degrees",
    28: "Pick a puck, bypass a wall and place the puck",
    29: "Pick up a puck from a hole",
    30: "Pick and place a puck to a goal",
    31: "Slide a plate into a cabinet",
    32: "Slide a plate into a cabinet sideways",
    33: "Get a plate from the cabinet",
    34: "Get a plate from the cabinet sideways",
    35: "Insert a peg sideways",
    36: "Unplug a peg sideways",
    37: "Kick a soccer into the goal",
    38: "Grasp a stick and push a box using the stick",
    39: "Grasp a stick and pull a box with the stick",
    40: "Push the puck to a goal",
    41: "Bypass a wall and push a puck to a goal",
    42: "Reach a goal position",
    43: "Bypass a wall and reach a goal",
    44: "Pick and place a puck onto a shelf",
    45: "Sweep a puck into a hole",
    46: "Sweep a puck off the table",
    47: "Push and open a window",
    48: "Push and close a window",
    49: "Push the puck back toward the robot to the goal",
}

# The goal is NAMED, never located (team decision): the model has to find it in the image.
# REVIEW: these names are a proposal; edit before building the training set, then re-freeze.
GOAL_NAMES = {
    0: "the peg", 1: "the basket", 2: "the empty bin", 3: "the box", 4: "the button",
    5: "the button", 6: "the button", 7: "the button", 8: "the coffee machine button",
    9: "the goal spot", 10: "the spot under the coffee machine", 11: "the dial",
    12: "the goal spot", 13: "the door", 14: "the door lock", 15: "the door",
    16: "the door lock", 17: "the hole", 18: "the drawer", 19: "the drawer",
    20: "the faucet handle", 21: "the faucet handle", 22: "the screw", 23: "the handle",
    24: "the handle", 25: "the handle", 26: "the handle", 27: "the lever",
    28: "the goal spot", 29: "the goal spot", 30: "the goal spot", 31: "the cabinet",
    32: "the cabinet", 33: "the goal spot", 34: "the goal spot", 35: "the hole",
    36: "the goal spot", 37: "the soccer goal", 38: "the goal spot", 39: "the goal spot",
    40: "the goal spot", 41: "the goal spot", 42: "the goal spot", 43: "the goal spot",
    44: "the shelf", 45: "the hole", 46: "the table edge", 47: "the window",
    48: "the window", 49: "the goal spot",
}

# ---------------------------------------------------------------- episodes
SEEDS_PER_TASK = 20                 # 5 tasks x 20 = 100 held-out episodes
SEED_START = 1000                   # eval seeds 1000..1019 (env built with this seed)
MAX_STEPS = 500                     # MetaWorld max_path_length; success = info["success"] at any step
CHUNK = 3                           # actions per model call, executed open-loop

# ---------------------------------------------------------------- observation
RENDER_SIZE = 480                   # same as the LeRobot MT50 images
IMAGE_SIZE = 256                    # resized (bicubic) like the training images
JPEG_QUALITY = 85
CAMERA = "corner2"
CAMERA_POS = (0.75, 0.075, 0.7)     # LeRobot MT50 corner2 pose (MetaWorld default is (1.3,-0.2,1.1))
GRIPPER_CLOSED_BELOW = 0.5          # "Gripper: closed" iff obs[3] < 0.5 (same rule as training data)

# ---------------------------------------------------------------- action codec
TOKEN_RANGE = 10                    # dx,dy,dz in -10..10, 10 == 1 cm (MetaWorld action 1.0)
GRIP_RANGE = 2                      # grip in -2..2 -> -1..1


def system_prompt(chunk: int = CHUNK) -> str:
    example = ";".join(["2,-1,0,0", "3,0,0,0", "3,0,0,1", "0,0,0,1", "0,0,0,0"][:chunk])
    return (
        "You are a robot control policy for a Sawyer arm on a tabletop. "
        "You see one camera frame, the task instruction, the goal, the current end-effector "
        "position in metres, and the gripper state. "
        f"Predict the next {chunk} actions. Each action is dx,dy,dz,grip where "
        "dx,dy,dz are integers from -10 to 10 meaning fractions of a centimetre of "
        "end-effector movement (10 = 1 cm), and grip is an integer from -2 to 2: "
        "2 closes the gripper hard, 0 holds the current grip, -2 opens it. "
        f"Reply with exactly {chunk} actions separated by semicolons and nothing else, "
        f"for example: {example}"
    )


def user_text(task_index: int, ee: list[float], closed: int, chunk: int = CHUNK) -> str:
    return (
        f"Task: {TASK_TEXT[task_index]}\n"
        f"Goal: {GOAL_NAMES[task_index]}\n"
        f"End effector: x={ee[0]:.3f}, y={ee[1]:.3f}, z={ee[2]:.3f}\n"
        f"Gripper: {'closed' if closed else 'open'}\n"
        f"Predict the next {chunk} actions."
    )


def messages(task_index: int, ee: list[float], closed: int, image_data_url: str) -> list[dict]:
    """The exact chat sent to every model (and used as the SFT prompt)."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": [
            {"type": "text", "text": user_text(task_index, ee, closed)},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]},
    ]


# Output language: xyz in {0, +-1..+-10}, grip in {0, +-1, +-2}, exactly CHUNK actions.
GRAMMAR = (
    'root ::= act ";" act ";" act\n'
    'act ::= xyz "," xyz "," xyz "," grip\n'
    'xyz ::= "0" | "-"? [1-9] | "-"? "10"\n'
    'grip ::= "0" | "-"? [1-2]\n'
)
_XYZ = r"(?:0|-?[1-9]|-?10)"
_ACT = rf"({_XYZ}),({_XYZ}),({_XYZ}),(0|-?[12])"
ACT_RE = re.compile(_ACT)
ANSWER_RE = re.compile(rf"{_ACT}(?:;{_ACT}){{{CHUNK - 1}}}")

# Decoding, identical for every model. The model id and the reasoning switch are the only
# per-model fields (see models.json).
DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "seed": 0,
    "response_format": {"type": "grammar", "grammar": GRAMMAR},
}

NOOP = [0, 0, 0, 0]


def parse_answer(text: str) -> tuple[list[list[int]], bool]:
    """Fixed rule for every model: strip whitespace; if the whole reply is a valid answer
    use its CHUNK actions. Otherwise use the valid actions found in order (at most CHUNK),
    padded with no-ops, and flag the reply invalid. Nothing else is ever executed."""
    t = (text or "").strip()
    if ANSWER_RE.fullmatch(t):
        acts = [[int(v) for v in m.groups()] for m in ACT_RE.finditer(t)]
        return acts, True
    acts = [[int(v) for v in m.groups()] for m in ACT_RE.finditer(t)][:CHUNK]
    return acts + [NOOP] * (CHUNK - len(acts)), False


def dequantize(tok: list[int]) -> list[float]:
    xyz = [max(-TOKEN_RANGE, min(TOKEN_RANGE, v)) / TOKEN_RANGE for v in tok[:3]]
    return xyz + [max(-GRIP_RANGE, min(GRIP_RANGE, tok[3])) / GRIP_RANGE]


def quantize(a) -> list[int]:
    """Inverse used for training targets and the quantized-expert ceiling."""
    xyz = [int(round(max(-1.0, min(1.0, float(v))) * TOKEN_RANGE)) for v in a[:3]]
    return xyz + [int(max(-GRIP_RANGE, min(GRIP_RANGE, round(float(a[3]) * GRIP_RANGE))))]


def spec_fingerprint() -> str:
    blob = json.dumps({
        "v": BENCH_VERSION, "test": TEST_TASKS, "text": TASK_TEXT, "goal": GOAL_NAMES,
        "seeds": [SEED_START, SEEDS_PER_TASK], "max_steps": MAX_STEPS, "chunk": CHUNK,
        "img": [RENDER_SIZE, IMAGE_SIZE, JPEG_QUALITY, CAMERA, CAMERA_POS],
        "grip": GRIPPER_CLOSED_BELOW, "sys": system_prompt(), "decoding": DECODING,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
