1. diffusion planner for pushT
2. diffusion for leRobotDataset

Both train a conditional 1D UNet that denoises an action sequence (Diffusion Policy). The observation history is FiLM conditioning. Training uses the `kaggle` conda env.

PushT uses the 5D state `[agent_x, agent_y, block_x, block_y, block_angle]` from `pusht_cchi_v7_replay.zarr.zip`. LeRobot training reads a v2 or v3 dataset (default `lerobot/pusht`) and conditions on every float `observation.*` column. On that dataset the float state is only the 2D pusher position; the block pose is in the video.

```bash
conda run -n kaggle python 01_pushT/train.py --epochs 100
conda run -n kaggle python 02_leRobot/train.py --repo-id lerobot/pusht --epochs 100
conda run -n kaggle python 03_leRobot/train.py --epochs 100
```

`03_leRobot` trains the same policy on ALOHA, a bimanual robot. State and action are 14 joint values, and the top camera is cropped onto the table so the policy can see the cube. The dataset is `lerobot/aloha_sim_transfer_cube_human`. The checkpoint is `03_leRobot/outputs/best.pt`.

Checkpoints go to `01_pushT/outputs/best.pt`, `02_leRobot/outputs/best.pt`, and `03_leRobot/outputs/best.pt`. `--max-steps` stops an otherwise full run early. `--obs-source keypoint` on PushT uses 9 keypoints plus the pusher position.

Roll a checkpoint out in the PushT simulator and write one mp4 per seed:

```bash
conda run -n kaggle python infer.py --checkpoint 01_pushT/outputs/best.pt
conda run -n kaggle python infer.py --checkpoint 02_leRobot/outputs/best.pt
MUJOCO_GL=egl conda run -n kaggle python 03_leRobot/infer.py
```
