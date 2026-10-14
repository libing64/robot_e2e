"""Train a diffusion planner on PushT state demonstrations.

Observation is the 5D simulator state ``[agent_x, agent_y, block_x, block_y, block_angle]``.
The policy denoises a future action sequence conditioned on a short observation history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_diffusion.train import run_training  # noqa: E402
from dataset import default_pusht_path, make_pusht_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="pusht zarr or zarr.zip")
    parser.add_argument("--obs-source", choices=("state", "keypoint"), default="state")
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
    dataset_path = args.dataset or default_pusht_path()
    if not dataset_path.is_file():
        raise SystemExit(
            f"PushT dataset not found: {dataset_path}\n"
            "Pass --dataset or set PUSHT_ZARR to pusht_cchi_v7_replay.zarr.zip"
        )
    print(f"dataset {dataset_path}  obs_source {args.obs_source}")
    train_set, val_set = make_pusht_datasets(
        dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        val_ratio=args.val_ratio,
        seed=args.seed,
        obs_source=args.obs_source,
    )
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
