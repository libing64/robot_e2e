"""Roll the ALOHA diffusion policy out in the transfer-cube simulator.

Each replan draws both gripper trajectories on the top camera. Red and orange
are the left arm (executed steps, then the rest of the horizon). Blue and cyan
are the right arm. A white ring marks the command being executed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_leRobot"))
sys.path.insert(0, str(ROOT))

from action_diffusion.policy import load_policy  # noqa: E402
from gym_aloha.constants import unnormalize_puppet_gripper_position  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402

LEFT_GRIPPER = "vx300s_left/gripper_link"
RIGHT_GRIPPER = "vx300s_right/gripper_link"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "best.pt",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--execute-steps", type=int, default=8,
                        help="How many predicted actions to run before replanning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "videos")
    return parser.parse_args()


def resize_camera(frame: np.ndarray, size: int) -> np.ndarray:
    from vision_data import CROP_H, CROP_W, CROP_X, CROP_Y

    cropped = frame[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def plan_actions(policy, obs_deque: deque, image_deque: deque | None, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    obs_seq = np.stack(obs_deque).astype(np.float32)
    obs_seq = torch.from_numpy(obs_seq).to(device=device).unsqueeze(0)
    image = None
    if image_deque is not None:
        frames = np.stack(image_deque)
        image = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    sample = policy.sample(obs_seq, image)
    executed = sample["action"][0].detach().cpu().numpy()
    predicted = sample["action_pred"][0].detach().cpu().numpy()
    return executed, predicted


def apply_arm_qpos(qpos: np.ndarray, action: np.ndarray) -> None:
    left_grip = unnormalize_puppet_gripper_position(float(np.clip(action[6], 0.0, 1.0)))
    right_grip = unnormalize_puppet_gripper_position(float(np.clip(action[13], 0.0, 1.0)))
    qpos[0:6] = action[0:6]
    qpos[6] = left_grip
    qpos[7] = -left_grip
    qpos[8:14] = action[7:13]
    qpos[14] = right_grip
    qpos[15] = -right_grip


def gripper_world(physics, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    saved = np.array(physics.data.qpos, copy=True)
    qpos = np.array(physics.data.qpos, copy=True)
    apply_arm_qpos(qpos, action)
    physics.data.qpos[:] = qpos
    physics.forward()
    left = np.array(physics.named.data.xpos[LEFT_GRIPPER], copy=True)
    right = np.array(physics.named.data.xpos[RIGHT_GRIPPER], copy=True)
    physics.data.qpos[:] = saved
    physics.forward()
    return left, right


def trajectory_world(physics, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for action in predicted:
        left_pos, right_pos = gripper_world(physics, action)
        left.append(left_pos)
        right.append(right_pos)
    return np.stack(left), np.stack(right)


def project(physics, point: np.ndarray, width: int, height: int, camera: str = "top"):
    cam_id = physics.model.name2id(camera, "camera")
    cam_pos = np.asarray(physics.data.cam_xpos[cam_id], dtype=np.float64)
    rotation = np.asarray(physics.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - cam_pos)
    if local[2] >= -1e-6:
        return None
    fovy = np.deg2rad(float(physics.model.cam_fovy[cam_id]))
    focal = 0.5 * height / np.tan(fovy / 2.0)
    u = width / 2.0 + focal * (local[0] / -local[2])
    v = height / 2.0 - focal * (local[1] / -local[2])
    return int(round(u)), int(round(v))


def draw_chain(image: np.ndarray, pixels: list[tuple[int, int] | None], color: tuple[int, int, int], radius: int) -> None:
    valid = [(index, pixel) for index, pixel in enumerate(pixels) if pixel is not None]
    if len(valid) >= 2:
        chain = np.asarray([pixel for _, pixel in valid], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [chain], False, color, 2, cv2.LINE_AA)
    for _, pixel in valid:
        cv2.circle(image, pixel, radius, color, -1, cv2.LINE_AA)


def draw_plan(image: np.ndarray, physics, left_world: np.ndarray, right_world: np.ndarray,
              execute_start: int, execute_count: int, cursor: int) -> np.ndarray:
    height, width = image.shape[:2]
    frame = np.ascontiguousarray(image)
    left_now = np.asarray(physics.named.data.xpos[LEFT_GRIPPER], dtype=np.float64)
    right_now = np.asarray(physics.named.data.xpos[RIGHT_GRIPPER], dtype=np.float64)
    execute_end = execute_start + execute_count

    def pixels(origin: np.ndarray, points: np.ndarray) -> list[tuple[int, int] | None]:
        projected = [project(physics, origin, width, height)]
        projected.extend(project(physics, point, width, height) for point in points)
        return projected

    def paint_arm(chain: list[tuple[int, int] | None], executed: tuple[int, int, int], horizon: tuple[int, int, int]) -> None:
        draw_chain(frame, chain, horizon, 4)
        # chain[0] is the live gripper; predicted waypoints start at chain[1].
        executed_chain = chain[:1] + chain[1 + execute_start:1 + execute_end]
        draw_chain(frame, executed_chain, executed, 6)
        current = chain[1 + execute_start + cursor]
        if current is not None:
            cv2.circle(frame, current, 8, (255, 255, 255), 2, cv2.LINE_AA)

    paint_arm(pixels(left_now, left_world), (220, 40, 40), (245, 170, 30))
    paint_arm(pixels(right_now, right_world), (40, 90, 230), (70, 210, 230))
    return frame


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    image = np.ascontiguousarray(frame)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "L red/orange   R blue/cyan", (12, image.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "L red/orange   R blue/cyan", (12, image.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def rollout(policy, seed: int, max_steps: int, width: int, height: int, device: torch.device, execute_steps: int):
    env = AlohaEnv(
        task="transfer_cube",
        obs_type="pixels_agent_pos",
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
    )
    obs, _info = env.reset(seed=seed)
    physics = env._env.physics
    obs_deque = deque([np.asarray(obs["agent_pos"], dtype=np.float32)] * policy.obs_horizon, maxlen=policy.obs_horizon)
    image_deque = None
    if policy.image_size:
        image_deque = deque(
            [resize_camera(env.render(), policy.image_size)] * policy.obs_horizon,
            maxlen=policy.obs_horizon,
        )
    execute_start = policy.obs_horizon - 1
    frames = []
    rewards = []
    success = False
    step = 0
    actions = np.zeros((0, policy.action_dim), dtype=np.float32)
    left_world = np.zeros((0, 3), dtype=np.float64)
    right_world = np.zeros((0, 3), dtype=np.float64)
    cursor = 0

    def paint(reward: float) -> None:
        image = draw_plan(
            env.render(), physics, left_world, right_world, execute_start, len(actions), cursor,
        )
        status = "success" if success else f"reward {reward:.0f}"
        label = f"seed {seed}  reward 0" if step == 0 else f"seed {seed}  step {step}  {status}"
        frames.append(annotate(image, label))

    while not success and step < max_steps:
        if cursor >= len(actions):
            actions, predicted = plan_actions(policy, obs_deque, image_deque, device)
            if execute_steps > 0:
                actions = actions[:execute_steps]
            left_world, right_world = trajectory_world(physics, predicted)
            cursor = 0
            if step == 0:
                paint(0.0)
        obs, reward, terminated, _truncated, info = env.step(actions[cursor])
        obs_deque.append(np.asarray(obs["agent_pos"], dtype=np.float32))
        if image_deque is not None:
            image_deque.append(resize_camera(env.render(), policy.image_size))
        rewards.append(float(reward))
        success = bool(info.get("is_success", terminated))
        step += 1
        paint(float(reward))
        cursor += 1
    env.close()
    return frames, rewards, success


def write_video(path: Path, frames: list[np.ndarray], fps: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
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
    if policy.obs_dim != 14 or policy.action_dim != 14:
        raise SystemExit(f"expected a 14-DoF ALOHA policy, got obs {policy.obs_dim} action {policy.action_dim}")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    print(f"checkpoint {args.checkpoint}")
    for seed in seeds:
        frames, rewards, success = rollout(
            policy, seed, args.max_steps, args.width, args.height, device, args.execute_steps,
        )
        path = args.output / f"seed_{seed}.mp4"
        write_video(path, frames, fps=50)
        peak = max(rewards) if rewards else 0.0
        print(f"seed {seed}  steps {len(rewards)}  peak_reward {peak:.0f}  success {success}  {path}")


if __name__ == "__main__":
    main()
