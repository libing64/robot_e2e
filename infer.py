"""Roll a trained diffusion policy out in PushT and write mp4 videos.

The 5D PushT checkpoint conditions on agent position, block position, and block
angle. The LeRobot checkpoint only sees the 2D agent position.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/libing/source/ml/diffusion_planner/diffusion_policy")

from action_diffusion.policy import load_policy  # noqa: E402
from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely  # noqa: E402


def block_coverage(env: PushTEnv) -> float:
    goal_body = env._get_goal_pose_body(env.goal_pose)
    goal_geom = pymunk_to_shapely(goal_body, env.block.shapes).buffer(0)
    block_geom = pymunk_to_shapely(env.block, env.block.shapes).buffer(0)
    return float(goal_geom.intersection(block_geom).area / goal_geom.area)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="100000,100001,100002")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--output", type=Path, default=None,
                        help="directory for mp4 files; default is <checkpoint_dir>/videos")
    return parser.parse_args()


def observation_for_policy(obs: np.ndarray, obs_dim: int) -> np.ndarray:
    if obs_dim == obs.shape[-1]:
        return obs
    if obs_dim == 2:
        return obs[:2]
    raise ValueError(f"policy obs_dim={obs_dim} does not match PushT observation {obs.shape}")


@torch.no_grad()
def plan_actions(policy, obs_deque: deque, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    obs_seq = np.stack([observation_for_policy(obs, policy.obs_dim) for obs in obs_deque])
    obs_seq = torch.from_numpy(obs_seq).to(device=device, dtype=torch.float32).unsqueeze(0)
    sample = policy.sample(obs_seq)
    executed = sample["action"][0].detach().cpu().numpy()
    predicted = sample["action_pred"][0].detach().cpu().numpy()
    return executed, predicted


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    image = np.ascontiguousarray(frame)
    cv2.putText(
        image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (20, 20, 20), 4, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return image


def draw_trajectory(
    frame: np.ndarray,
    agent_xy: np.ndarray,
    predicted: np.ndarray,
    execute_start: int,
    execute_count: int,
    cursor: int,
    render_size: int,
    workspace: float = 512.0,
) -> np.ndarray:
    """Draw one diffusion plan. Red points are executed; orange points are the rest of the horizon."""
    image = np.ascontiguousarray(frame)
    scale = render_size / workspace
    points = np.round(np.asarray(predicted, dtype=np.float32) * scale).astype(np.int32)
    agent = np.round(np.asarray(agent_xy, dtype=np.float32) * scale).astype(np.int32)
    chain = np.vstack([agent.reshape(1, 2), points])
    cv2.polylines(image, [chain], False, (230, 120, 20), 2, cv2.LINE_AA)
    execute_end = execute_start + execute_count
    for index, point in enumerate(points):
        xy = (int(point[0]), int(point[1]))
        if execute_start <= index < execute_end:
            cv2.circle(image, xy, 5, (220, 40, 40), -1, cv2.LINE_AA)
        else:
            cv2.circle(image, xy, 3, (245, 170, 30), -1, cv2.LINE_AA)
    current = points[execute_start + cursor]
    cv2.circle(image, (int(current[0]), int(current[1])), 8, (170, 0, 180), 2, cv2.LINE_AA)
    return image


def rollout(policy, seed: int, max_steps: int, render_size: int, device: torch.device):
    env = PushTEnv(render_size=render_size, render_action=False)
    env.seed(seed)
    obs = env.reset()
    obs_deque = deque([obs] * policy.obs_horizon, maxlen=policy.obs_horizon)
    execute_start = policy.obs_horizon - 1
    frames = []
    coverages = []
    done = False
    step = 0
    actions = np.zeros((0, 2), dtype=np.float32)
    predicted = np.zeros((0, 2), dtype=np.float32)
    cursor = 0

    def paint(coverage: float) -> None:
        frame = draw_trajectory(
            env.render("rgb_array"),
            np.asarray(env.agent.position, dtype=np.float32),
            predicted,
            execute_start,
            len(actions),
            cursor,
            render_size,
        )
        status = "success" if done else f"coverage {coverage:.2f}"
        label = f"seed {seed}  coverage 0.00" if step == 0 else f"seed {seed}  step {step}  {status}"
        frames.append(annotate(frame, label))

    while not done and step < max_steps:
        if cursor >= len(actions):
            actions, predicted = plan_actions(policy, obs_deque, device)
            cursor = 0
            if step == 0:
                paint(0.0)
        obs, _reward, done, _info = env.step(actions[cursor])
        obs_deque.append(obs)
        coverage = block_coverage(env)
        coverages.append(coverage)
        step += 1
        paint(coverage)
        cursor += 1
    env.close()
    return frames, coverages, done


def write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"frame size {width}x{height} must be even for yuv420p")
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device)
    out_dir = args.output or (args.checkpoint.parent / "videos")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    print(
        f"checkpoint {args.checkpoint}  obs_dim={policy.obs_dim}  "
        f"action_horizon={policy.action_horizon}"
    )
    for seed in seeds:
        frames, coverages, done = rollout(policy, seed, args.max_steps, args.render_size, device)
        path = out_dir / f"seed_{seed}.mp4"
        write_video(path, frames)
        coverage = max(coverages) if coverages else 0.0
        print(f"seed {seed}  steps {len(coverages)}  coverage {coverage:.3f}  success {bool(done)}  {path}")


if __name__ == "__main__":
    main()
