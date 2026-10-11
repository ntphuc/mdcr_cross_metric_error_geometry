from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .model import ConvBlock, DecoderBlock, UNetEfficientNetB0


class ResNet18Encoder(nn.Module):
    """Feature pyramid compatible with a compact U-Net decoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            m = resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(f"Could not load pretrained ResNet18 weights ({exc}); using random init.")
            m = resnet18(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)  # 64 @ 1/2
        self.pool = m.maxpool
        self.layer1 = m.layer1  # 64 @ 1/4
        self.layer2 = m.layer2  # 128 @ 1/8
        self.layer3 = m.layer3  # 256 @ 1/16
        self.layer4 = m.layer4  # 512 @ 1/32

    def forward(self, x: torch.Tensor):
        s1 = self.stem(x)
        x = self.pool(s1)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return s1, s2, s3, s4, s5


class UNetResNet18(nn.Module):
    """U-Net-style decoder with a torchvision ResNet18 encoder."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBlock(512, 256)
        self.dec4 = DecoderBlock(256, 256, 128)
        self.dec3 = DecoderBlock(128, 128, 64)
        self.dec2 = DecoderBlock(64, 64, 40)
        self.dec1 = DecoderBlock(40, 64, 24)
        self.head = nn.Sequential(ConvBlock(24, 24), nn.Conv2d(24, 1, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        x = self.bottleneck(s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.head(x)


def canonical_arch(name: str | None) -> str:
    x = str(name or "effb0_unet").strip().lower().replace("-", "_")
    aliases = {
        "efficientnet_b0": "effb0_unet",
        "efficientnetb0": "effb0_unet",
        "unet_effb0": "effb0_unet",
        "effb0": "effb0_unet",
        "resnet18": "resnet18_unet",
        "unet_resnet18": "resnet18_unet",
        "res18_unet": "resnet18_unet",
    }
    return aliases.get(x, x)


def build_segmentor(architecture: str | None = None, pretrained_encoder: bool = True) -> nn.Module:
    arch = canonical_arch(architecture)
    if arch == "effb0_unet":
        return UNetEfficientNetB0(pretrained_encoder=pretrained_encoder)
    if arch == "resnet18_unet":
        return UNetResNet18(pretrained_encoder=pretrained_encoder)
    raise ValueError(f"Unknown architecture={architecture!r}. Supported: effb0_unet, resnet18_unet")


def architecture_from_config(cfg: dict) -> str:
    return canonical_arch(cfg.get("architecture", "effb0_unet"))
