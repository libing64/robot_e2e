"""Top-camera frames for the ALOHA cube-transfer dataset."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from action_diffusion.windows import WindowDataset, split_episode_mask
from dataset import load_lerobot_arrays, resolve_root

IMAGE_SIZE = 224
# Table region that contains the cube. Full 640x480 frames shrink the cube to a few pixels.
CROP_X, CROP_Y, CROP_W, CROP_H = 240, 180, 240, 200


def decode_top_camera(video: Path, image_size: int, n_frames: int) -> np.ndarray:
    process = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-i", str(video),
            "-vf", f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y},scale={image_size}:{image_size}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    raw = process.stdout.read()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while decoding {video}")
    frame_bytes = image_size * image_size * 3
    if len(raw) != n_frames * frame_bytes:
        raise RuntimeError(
            f"decoded {len(raw) // frame_bytes} frames from {video}, expected {n_frames}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, image_size, image_size, 3)


def load_top_images(root: Path, n_frames: int, image_size: int = IMAGE_SIZE) -> np.ndarray:
    cache = root / f"top_{image_size}.npy"
    if cache.is_file():
        images = np.load(cache, mmap_mode="r")
        if images.shape == (n_frames, image_size, image_size, 3):
            return images
    video = root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"missing top camera video {video}")
    print(f"decoding {video} to {image_size}x{image_size}")
    images = decode_top_camera(video, image_size, n_frames)
    np.save(cache, images)
    return np.load(cache, mmap_mode="r")


def make_aloha_datasets(
    repo_or_path: str = "lerobot/aloha_sim_transfer_cube_human",
    *,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    val_ratio: float = 0.02,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> tuple[WindowDataset, WindowDataset, Path]:
    root = resolve_root(repo_or_path)
    obs, action, episode_ends, _keys = load_lerobot_arrays(root, ["observation.state"], "action")
    images = load_top_images(root, len(obs), image_size)
    train_mask = split_episode_mask(len(episode_ends), val_ratio, seed)
    common = dict(
        obs=obs,
        action=action,
        episode_ends=episode_ends,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        images=images,
    )
    train_set = WindowDataset(episode_mask=train_mask, **common)
    val_set = WindowDataset(episode_mask=~train_mask, **common)
    return train_set, val_set, root
