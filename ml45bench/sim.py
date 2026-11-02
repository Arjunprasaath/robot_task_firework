"""MetaWorld side of the benchmark: build an episode, render the model's observation, step.

Deterministic by construction: an episode is fully defined by (env name, seed). MetaWorld
MT1 fixes object/goal positions when the env is *built* (reset(seed) does not move them),
so every episode builds a fresh env with its own seed.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")

import gymnasium as gym  # noqa: E402
import metaworld  # noqa: E402,F401  (registers Meta-World/MT1)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from . import spec  # noqa: E402


def make_env(env_name: str, seed: int, camera_pos=spec.CAMERA_POS):
    """Env without gymnasium's renderer: frames come from a per-episode mujoco.Renderer
    (gymnasium's glfw viewer breaks other envs' contexts in the same process on close)."""
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
    m, d = env.unwrapped.model, env.unwrapped.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, spec.CAMERA)
    m.cam_pos[cid] = camera_pos
    mujoco.mj_forward(m, d)  # camera pose only takes effect after a forward pass
    return env


def frame(env, renderer) -> np.ndarray:
    """Raw 480x480 RGB from the corner2 camera, rotated 180 deg like the LeRobot MT50 images."""
    renderer.update_scene(env.unwrapped.data, camera=spec.CAMERA)
    return np.ascontiguousarray(np.flip(renderer.render(), (0, 1)))


def jpeg(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb).resize((spec.IMAGE_SIZE, spec.IMAGE_SIZE), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=spec.JPEG_QUALITY)
    return buf.getvalue()


def data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def proprio(obs) -> tuple[list[float], int]:
    """observation.state in the dataset == obs[:4] == [ee_x, ee_y, ee_z, gripper_open]."""
    ee = [round(float(x), 3) for x in obs[:3]]
    return ee, int(float(obs[3]) < spec.GRIPPER_CLOSED_BELOW)


def scene(obs) -> dict:
    """Initial object / goal position: logged per episode, checked against the manifest."""
    return {"obj": [round(float(x), 4) for x in obs[4:7]],
            "goal": [round(float(x), 4) for x in obs[-3:]]}


class Episode:
    """One held-out episode. Call `observe()` for the model input, `step(actions)` to apply."""

    def __init__(self, task_index: int, seed: int):
        self.task_index, self.seed = task_index, seed
        self.env = make_env(spec.TEST_TASKS.get(task_index) or _env_name(task_index), seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.renderer = mujoco.Renderer(self.env.unwrapped.model, spec.RENDER_SIZE, spec.RENDER_SIZE)
        self.t = 0
        self.success_step: int | None = None
        self.max_reward = 0.0
        self.min_obj_to_target = float("inf")
        self.grasped = False
        self.frames: list[np.ndarray] = []

    @property
    def done(self) -> bool:
        return self.success_step is not None or self.t >= spec.MAX_STEPS

    def observe(self, keep_frame: bool = False) -> tuple[bytes, list[float], int]:
        rgb = frame(self.env, self.renderer)
        if keep_frame:
            self.frames.append(rgb)
        ee, closed = proprio(self.obs)
        return jpeg(rgb), ee, closed

    def step(self, actions: list[list[int]]) -> None:
        for tok in actions:
            if self.done:
                break
            self.obs, r, _, _, info = self.env.step(np.asarray(spec.dequantize(tok), dtype=np.float64))
            self.t += 1
            self.max_reward = max(self.max_reward, float(r))
            self.min_obj_to_target = min(self.min_obj_to_target, float(info.get("obj_to_target", np.inf)))
            self.grasped = self.grasped or bool(info.get("grasp_success", 0.0) > 0)
            if info.get("success", 0.0) > 0 and self.success_step is None:
                self.success_step = self.t

    def close(self):
        self.renderer.close()
        self.env.close()


def _env_name(task_index: int) -> str:
    raise KeyError(f"no env name registered for task {task_index}")
