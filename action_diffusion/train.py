"""Shared optimization loop for the action diffusion policy."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from action_diffusion.policy import DiffusionPolicy, EMA, save_policy, stats_from_frames


def _cosine_warmup_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_training(
    train_set,
    val_set,
    out_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    num_train_timesteps: int = 100,
    num_inference_steps: int = 16,
    seed: int = 0,
    device: str = "cuda",
    max_steps: int = 0,
    num_workers: int = 0,
    down_dims: list[int] | None = None,
    warmup_steps: int = 500,
) -> Path:
    if down_dims is None:
        down_dims = [256, 512, 1024]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device_t = torch.device(device)

    sample = train_set[0]
    obs_dim = int(sample["obs"].shape[-1])
    action_dim = int(sample["action"].shape[-1])
    stats = stats_from_frames(train_set.obs_frames, train_set.action_frames)

    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=train_set.obs_horizon,
        pred_horizon=train_set.pred_horizon,
        action_horizon=train_set.action_horizon,
        down_dims=down_dims,
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
    )
    policy.set_normalizer(stats["obs_min"], stats["obs_max"], stats["action_min"], stats["action_max"])
    policy.to(device_t)

    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"obs_dim={obs_dim} action_dim={action_dim} "
        f"train_windows={len(train_set)} val_windows={len(val_set)} params={n_params/1e6:.2f}M"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
        drop_last=len(train_set) > batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup = min(warmup_steps, max(total_steps // 10, 1))
    ema = EMA(policy)
    ema_policy = copy.deepcopy(policy)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        **policy.export_config(),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "warmup_steps": warmup,
    }, indent=2))

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    global_step = 0
    policy.train()
    started = time.time()

    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            obs = batch["obs"].to(device_t, non_blocking=True)
            action = batch["action"].to(device_t, non_blocking=True)
            loss = policy.loss(obs, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            step_lr = lr * _cosine_warmup_lambda(global_step, total_steps, warmup)
            for group in optimizer.param_groups:
                group["lr"] = step_lr
            optimizer.step()
            ema.update(policy)
            losses.append(loss.item())
            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                break

        ema.copy_to(ema_policy)
        val_loss = _evaluate(ema_policy, val_loader, device_t)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"epoch {epoch + 1:03d}  step {global_step}  "
            f"train {train_loss:.5f}  val {val_loss:.5f}  lr {step_lr:.2e}"
        )
        payload_extra = {"epoch": epoch + 1, "step": global_step, "val_loss": val_loss}
        save_policy(out_dir / "last.pt", ema_policy, payload_extra)
        if val_loss < best_val:
            best_val = val_loss
            save_policy(best_path, ema_policy, payload_extra)
        if max_steps > 0 and global_step >= max_steps:
            break

    elapsed = time.time() - started
    print(f"done in {elapsed:.1f}s  best_val {best_val:.5f}  checkpoint {best_path}")
    _print_sample(ema_policy, val_set, device_t)
    return best_path


@torch.no_grad()
def _evaluate(model: DiffusionPolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        loss = model.loss(obs, action)
        total += loss.item() * obs.shape[0]
        count += obs.shape[0]
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def _print_sample(model: DiffusionPolicy, dataset, device: torch.device) -> None:
    model.eval()
    batch = dataset[0]
    obs = batch["obs"].unsqueeze(0).to(device)
    pred = model.sample(obs)
    action = pred["action"][0, 0].detach().cpu().numpy()
    target = batch["action"][model.obs_horizon - 1].numpy()
    print("sample action[0]", np.array2string(action, precision=2, suppress_small=True))
    print("demo action     ", np.array2string(target, precision=2, suppress_small=True))
