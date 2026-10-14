"""Train action diffusion on a LeRobot dataset.

Float observation features condition a 1D UNet that denoises the action
sequence. For ``lerobot/pusht`` the only float observation is the 2D pusher
position; the block pose lives in ``observation.image``. The 5D state planner
is ``01_pushT``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_diffusion.train import run_training  # noqa: E402
from dataset import make_lerobot_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default="lerobot/pusht",
                        help="Hugging Face dataset id or a local LeRobot dataset directory")
    parser.add_argument("--obs-keys", type=str, default="",
                        help="comma-separated float observation columns; default is every observation.* float")
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--down-dims", type=str, default="256,512,1024")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps; 0 = full run")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs_keys = [key for key in args.obs_keys.split(",") if key] or None
    train_set, val_set, root, used_keys = make_lerobot_datasets(
        args.repo_id,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        val_ratio=args.val_ratio,
        seed=args.seed,
        obs_keys=obs_keys,
        action_key=args.action_key,
    )
    print(f"dataset {root}")
    print(f"obs_keys {used_keys}  action_key {args.action_key}")
    run_training(
        train_set,
        val_set,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        down_dims=[int(x) for x in args.down_dims.split(",") if x],
    )


if __name__ == "__main__":
    main()
