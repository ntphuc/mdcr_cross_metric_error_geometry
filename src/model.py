from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EfficientNetB0Encoder(nn.Module):
    """Feature pyramid from torchvision EfficientNet-B0."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained EfficientNet-B0 weights ({exc}); "
                "falling back to random initialization."
            )
            backbone = efficientnet_b0(weights=None)

        f = backbone.features
        self.stage1 = nn.Sequential(f[0], f[1])      # 16  @ 1/2
        self.stage2 = f[2]                           # 24  @ 1/4
        self.stage3 = f[3]                           # 40  @ 1/8
        self.stage4 = nn.Sequential(f[4], f[5])      # 112 @ 1/16
        self.stage5 = nn.Sequential(f[6], f[7])      # 320 @ 1/32

    def forward(self, x: torch.Tensor):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s1, s2, s3, s4, s5


class UNetEfficientNetB0(nn.Module):
    """Simple U-Net decoder with EfficientNet-B0 encoder."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBlock(320, 256)
        self.dec4 = DecoderBlock(256, 112, 128)
        self.dec3 = DecoderBlock(128, 40, 64)
        self.dec2 = DecoderBlock(64, 24, 40)
        self.dec1 = DecoderBlock(40, 16, 24)
        self.head = nn.Sequential(
            ConvBlock(24, 24),
            nn.Conv2d(24, 1, kernel_size=1),
        )

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
