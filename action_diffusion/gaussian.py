"""Cosine DDPM training noise and DDIM sampling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Squared-cosine schedule used by Diffusion Policy (``squaredcos_cap_v2``)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(max=max_beta).to(dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    def __init__(self, num_train_timesteps: int = 100):
        super().__init__()
        betas = cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return s1 * x0 + s2 * noise

    def inference_timesteps(self, steps: int) -> torch.Tensor:
        steps = max(1, min(int(steps), self.num_train_timesteps))
        ts = torch.linspace(0, self.num_train_timesteps - 1, steps, device=self.alphas_cumprod.device)
        ts = ts.round().to(dtype=torch.long).unique(sorted=True)
        return torch.flip(ts, dims=(0,))

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        global_cond: torch.Tensor,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) from N(0, I) back to x0. Actions are clamped to [-1, 1]."""
        x = torch.randn(shape, device=device)
        timesteps = self.inference_timesteps(steps)
        alphas = self.alphas_cumprod
        for i, t in enumerate(timesteps):
            t_int = int(t.item())
            t_batch = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            eps = model(x, t_batch, global_cond)
            alpha_bar = alphas[t_int]
            alpha_bar_prev = alphas[int(timesteps[i + 1].item())] if i + 1 < len(timesteps) else x.new_tensor(1.0)
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar).clamp(min=1e-8)
            x0 = x0.clamp(-1.0, 1.0)
            x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
        return x
