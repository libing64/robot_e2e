"""PushT low-dimensional demonstrations for the diffusion planner."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import zarr

from action_diffusion.windows import WindowDataset, split_episode_mask


def default_pusht_path() -> Path:
    env = os.environ.get("PUSHT_ZARR")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "data" / "pusht_cchi_v7_replay.zarr.zip",
        Path("/home/libing/source/ml/diffusion_planner/diffusion_policy/pusht_cchi_v7_replay.zarr.zip"),
    ])
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_pusht_arrays(path: Path, obs_source: str = "state") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(str(path), mode="r")
    state = np.asarray(root["data"]["state"], dtype=np.float32)
    action = np.asarray(root["data"]["action"], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    if obs_source == "state":
        # [agent_x, agent_y, block_x, block_y, block_angle]
        obs = state
    elif obs_source == "keypoint":
        keypoint = np.asarray(root["data"]["keypoint"], dtype=np.float32).reshape(len(state), -1)
        obs = np.concatenate([keypoint, state[:, :2]], axis=-1)
    else:
        raise ValueError(f"unknown obs_source {obs_source!r}, expected 'state' or 'keypoint'")
    return obs, action, episode_ends


def make_pusht_datasets(
    path: Path,
    *,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    val_ratio: float = 0.02,
    seed: int = 42,
    obs_source: str = "state",
) -> tuple[WindowDataset, WindowDataset]:
    obs, action, episode_ends = load_pusht_arrays(path, obs_source)
    train_mask = split_episode_mask(len(episode_ends), val_ratio, seed)
    common = dict(
        obs=obs,
        action=action,
        episode_ends=episode_ends,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    train_set = WindowDataset(episode_mask=train_mask, **common)
    val_set = WindowDataset(episode_mask=~train_mask, **common)
    return train_set, val_set
