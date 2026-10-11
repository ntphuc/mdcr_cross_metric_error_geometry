from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import UNetEfficientNetB0


def _sobel_edge(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable normalized edge magnitude for a single-channel probability map."""
    dtype, device = prob.dtype, prob.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    gx = F.conv2d(prob, kx, padding=1)
    gy = F.conv2d(prob, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    # Sobel on [0,1] maps can exceed one. A bounded map is easier for the refiner.
    return mag / (1.0 + mag)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, groups: int = 8):
        padding = dilation
        gn_groups = max(1, min(groups, out_ch))
        while out_ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(inplace=True),
        )


class ResidualDilatedBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch, ch, dilation=dilation),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, ch)), ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class BoundaryResidualRefiner(nn.Module):
    """Small zero-initialized residual-logit head.

    Inputs are the normalized MR image, the current base probability, and a base
    boundary map. The final layer is initialized to zero, so epoch-0 deployment
    is exactly the source baseline rather than an untrained perturbation.
    """

    def __init__(self, width: int = 32, residual_scale: float = 2.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = ConvGNAct(5, width)
        self.blocks = nn.Sequential(
            ResidualDilatedBlock(width, 1),
            ResidualDilatedBlock(width, 2),
            ResidualDilatedBlock(width, 4),
            ResidualDilatedBlock(width, 2),
            ResidualDilatedBlock(width, 1),
        )
        self.out = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        base_prob = torch.sigmoid(base_logits)
        base_edge = _sobel_edge(base_prob)
        x = torch.cat([image, base_prob, base_edge], dim=1)
        raw = self.out(self.blocks(self.stem(x)))
        return self.residual_scale * torch.tanh(raw)


class EMBRModel(nn.Module):
    """EfficientNet-B0 U-Net plus a lightweight boundary residual refiner."""

    def __init__(
        self,
        base: UNetEfficientNetB0 | None = None,
        refiner_width: int = 32,
        residual_scale: float = 2.0,
    ):
        super().__init__()
        self.base = base if base is not None else UNetEfficientNetB0(pretrained_encoder=False)
        self.refiner = BoundaryResidualRefiner(refiner_width, residual_scale)

    def forward(self, image: torch.Tensor, return_parts: bool = False):
        base_logits = self.base(image)
        residual_logits = self.refiner(image, base_logits)
        final_logits = base_logits + residual_logits
        if return_parts:
            return final_logits, base_logits, residual_logits
        return final_logits

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)

    def unfreeze_late_base(self, modules: tuple[str, ...] = ("dec1", "head")) -> None:
        # Keep the encoder and early decoder frozen. Only the named late modules
        # are allowed to move at a low learning rate in the optional joint phase.
        for p in self.base.parameters():
            p.requires_grad_(False)
        for name in modules:
            mod = getattr(self.base, name, None)
            if mod is None:
                raise ValueError(f"Unknown base module for late unfreezing: {name}")
            for p in mod.parameters():
                p.requires_grad_(True)

    def set_phase_train_mode(self, joint: bool, modules: tuple[str, ...] = ("dec1", "head")) -> None:
        # Frozen BatchNorm statistics must not drift during refiner-only training.
        self.train()
        self.base.eval()
        self.refiner.train()
        if joint:
            for name in modules:
                getattr(self.base, name).train()
