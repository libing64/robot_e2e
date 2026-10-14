"""Observation-conditioned diffusion policy over action sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from action_diffusion.gaussian import GaussianDiffusion
from action_diffusion.unet1d import ConditionalUnet1D
from action_diffusion.vision import ImageEncoder


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
        down_dims: tuple[int, ...] | list[int] = (256, 512, 1024),
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        kernel_size: int = 5,
        image_size: int = 0,
        image_feature_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.down_dims = list(down_dims)
        self.image_size = image_size
        self.image_feature_dim = image_feature_dim
        self.encoder = ImageEncoder(image_feature_dim, image_size) if image_size else None
        cond_dim = obs_dim * obs_horizon
        if image_size:
            cond_dim += image_feature_dim * obs_horizon
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            down_dims=self.down_dims,
            kernel_size=kernel_size,
        )
        self.diffusion = GaussianDiffusion(num_train_timesteps)
        self.register_buffer("obs_min", torch.zeros(obs_dim))
        self.register_buffer("obs_max", torch.ones(obs_dim))
        self.register_buffer("action_min", torch.zeros(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))

    def set_normalizer(self, obs_min, obs_max, action_min, action_max) -> None:
        self.obs_min.copy_(torch.as_tensor(obs_min, dtype=torch.float32))
        self.obs_max.copy_(torch.as_tensor(obs_max, dtype=torch.float32))
        self.action_min.copy_(torch.as_tensor(action_min, dtype=torch.float32))
        self.action_max.copy_(torch.as_tensor(action_max, dtype=torch.float32))

    def _normalize(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return (x - lo) / (hi - lo) * 2.0 - 1.0

    def _unnormalize(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) / 2.0 * (hi - lo) + lo

    def _encode_images(self, image: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise ValueError("this checkpoint has no image encoder")
        if image.ndim != 5 or image.shape[1] < self.obs_horizon:
            raise ValueError(
                f"expected image (B, >= {self.obs_horizon}, 3, H, W), got {tuple(image.shape)}"
            )
        frames = image[:, : self.obs_horizon]
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        batch, steps, channels, height, width = frames.shape
        if channels != 3 or height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected image (B, T, 3, {self.image_size}, {self.image_size}), got {tuple(image.shape)}"
            )
        flat = frames.reshape(batch * steps, channels, height, width)
        flat = flat * 2.0 - 1.0
        features = self.encoder(flat).reshape(batch, steps, self.image_feature_dim)
        return features.flatten(1)

    def _global_cond(self, obs: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[1] < self.obs_horizon or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected obs (B, >= {self.obs_horizon}, {self.obs_dim}), got {tuple(obs.shape)}"
            )
        nobs = self._normalize(obs[:, : self.obs_horizon], self.obs_min, self.obs_max)
        cond = nobs.flatten(1)
        if self.encoder is not None:
            cond = torch.cat((cond, self._encode_images(image)), dim=-1)
        return cond

    def loss(self, obs: torch.Tensor, action: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        naction = self._normalize(action, self.action_min, self.action_max)
        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.diffusion.num_train_timesteps, (naction.shape[0],), device=naction.device
        )
        noisy = self.diffusion.add_noise(naction, noise, timesteps)
        pred = self.unet(noisy, timesteps, self._global_cond(obs, image))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, image: torch.Tensor | None = None, steps: int | None = None) -> dict[str, torch.Tensor]:
        """Denoise an action sequence. ``action`` is the slice the controller should execute."""
        steps = self.num_inference_steps if steps is None else steps
        cond = self._global_cond(obs, image)
        naction = self.diffusion.ddim_sample(
            self.unet,
            shape=(obs.shape[0], self.pred_horizon, self.action_dim),
            global_cond=cond,
            steps=steps,
            device=obs.device,
        )
        action_pred = self._unnormalize(naction, self.action_min, self.action_max)
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    def export_config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
            "action_horizon": self.action_horizon,
            "down_dims": self.down_dims,
            "num_train_timesteps": self.diffusion.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size,
            "image_feature_dim": self.image_feature_dim,
        }


class EMA:
    """Power-law EMA, matching the Diffusion Policy training recipe (power 0.75)."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_decay: float = 0.9999):
        self.power = power
        self.max_decay = max_decay
        self.step_count = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        decay = min(1.0 - (1.0 + self.step_count) ** (-self.power), self.max_decay)
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(self.shadow[name])


def save_policy(path: Path, model: DiffusionPolicy, extra: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "config": model.export_config(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy(path: Path | str, device: str | torch.device = "cpu") -> DiffusionPolicy:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    policy = DiffusionPolicy(**ckpt["config"])
    policy.load_state_dict(ckpt["model"])
    policy.to(device)
    policy.eval()
    return policy


def stats_from_frames(obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    def _limits(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
        lo = flat.min(axis=0)
        hi = np.maximum(flat.max(axis=0), lo + 1e-6)
        return lo, hi

    obs_min, obs_max = _limits(obs)
    action_min, action_max = _limits(action)
    return {
        "obs_min": obs_min,
        "obs_max": obs_max,
        "action_min": action_min,
        "action_max": action_max,
    }
