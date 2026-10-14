"""Episode-aware windows for receding-horizon action diffusion."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def split_episode_mask(n_episodes: int, val_ratio: float, seed: int) -> np.ndarray:
    """True marks training episodes. At least one episode stays in train."""
    mask = np.ones(n_episodes, dtype=bool)
    if n_episodes <= 1 or val_ratio <= 0:
        return mask
    n_val = int(round(n_episodes * val_ratio))
    n_val = min(max(n_val, 1), n_episodes - 1)
    val_idx = np.random.default_rng(seed).choice(n_episodes, size=n_val, replace=False)
    mask[val_idx] = False
    return mask


def build_window_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
    episode_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of fixed-length windows, with edge frames repeated as padding.

    Each row is ``(buffer_start, buffer_end, out_start, out_end)`` into the
    concatenated episode array and into the padded window.
    """
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    rows: list[list[int]] = []
    start_idx = 0
    for i, end_idx in enumerate(episode_ends):
        if episode_mask is not None and not episode_mask[i]:
            start_idx = int(end_idx)
            continue
        episode_length = int(end_idx) - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + start_idx
            buffer_end = min(idx + sequence_length, episode_length) + start_idx
            out_start = buffer_start - (idx + start_idx)
            out_end = sequence_length - ((idx + sequence_length + start_idx) - buffer_end)
            rows.append([buffer_start, buffer_end, out_start, out_end])
        start_idx = int(end_idx)
    if not rows:
        raise ValueError(
            "No training windows. Episodes are shorter than the prediction horizon "
            "after padding, or the split removed every episode."
        )
    return np.asarray(rows, dtype=np.int64)


def _pad_window(array: np.ndarray, buffer_start: int, buffer_end: int,
                out_start: int, out_end: int, length: int) -> np.ndarray:
    piece = array[buffer_start:buffer_end]
    if out_start == 0 and out_end == length:
        return piece
    window = np.empty((length,) + array.shape[1:], dtype=array.dtype)
    if out_start > 0:
        window[:out_start] = piece[0]
    if out_end < length:
        window[out_end:] = piece[-1]
    window[out_start:out_end] = piece
    return window


class WindowDataset(Dataset):
    """Samples ``obs`` of length obs_horizon and ``action`` of length pred_horizon."""

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        episode_ends: np.ndarray,
        episode_mask: np.ndarray,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        images: np.ndarray | None = None,
    ):
        if len(obs) != len(action):
            raise ValueError(f"obs/action length mismatch: {len(obs)} vs {len(action)}")
        if episode_ends[-1] != len(obs):
            raise ValueError(
                f"episode_ends[-1]={episode_ends[-1]} does not match frame count {len(obs)}"
            )
        if images is not None and len(images) != len(obs):
            raise ValueError(f"image/obs length mismatch: {len(images)} vs {len(obs)}")
        self.obs = np.asarray(obs, dtype=np.float32)
        self.action = np.asarray(action, dtype=np.float32)
        self.images = images
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.indices = build_window_indices(
            episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
            episode_mask=episode_mask,
        )
        frame_mask = np.zeros(len(self.obs), dtype=bool)
        start = 0
        for i, end in enumerate(np.asarray(episode_ends, dtype=np.int64)):
            if episode_mask[i]:
                frame_mask[start:end] = True
            start = int(end)
        self.obs_frames = self.obs[frame_mask]
        self.action_frames = self.action[frame_mask]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        buffer_start, buffer_end, out_start, out_end = self.indices[idx]
        obs = _pad_window(
            self.obs, int(buffer_start), int(buffer_end),
            int(out_start), int(out_end), self.pred_horizon,
        )[: self.obs_horizon]
        action = _pad_window(
            self.action, int(buffer_start), int(buffer_end),
            int(out_start), int(out_end), self.pred_horizon,
        )
        item = {
            "obs": torch.from_numpy(np.ascontiguousarray(obs)),
            "action": torch.from_numpy(np.ascontiguousarray(action)),
        }
        if self.images is not None:
            image = _pad_window(
                self.images, int(buffer_start), int(buffer_end),
                int(out_start), int(out_end), self.pred_horizon,
            )[: self.obs_horizon]
            item["image"] = torch.from_numpy(np.array(image, copy=True)).permute(0, 3, 1, 2)
        return item
