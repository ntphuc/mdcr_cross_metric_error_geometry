from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discordance import ROUTES
from .embr_model import ConvGNAct, ResidualDilatedBlock, _sobel_edge


def _binary_entropy_torch(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.float().clamp(float(eps), 1.0 - float(eps))
    q = (1.0 - p).clamp_min(float(eps))
    ent = -(p * torch.log(p) + q * torch.log(q))
    return torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)


def inference_cue_maps(base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ground-truth-free router/expert cue maps, always computed in FP32."""
    z = torch.nan_to_num(base_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    prob = torch.sigmoid(z).clamp(1e-6, 1.0 - 1e-6)
    edge = _sobel_edge(prob)
    entropy = (_binary_entropy_torch(prob) / 0.6931471805599453).clamp(0.0, 1.0)
    lowconf = (1.0 - 2.0 * torch.abs(prob - 0.5)).clamp(0.0, 1.0)
    return tuple(torch.nan_to_num(t.float(), nan=0.0, posinf=1.0, neginf=0.0) for t in (prob, edge, entropy, lowconf))


def inference_scalar_cues(prob: torch.Tensor, edge: torch.Tensor, entropy: torch.Tensor, lowconf: torch.Tensor) -> torch.Tensor:
    b, _, h, w = prob.shape
    dtype, device = prob.dtype, prob.device
    fg_soft = prob
    fg_hard = (prob >= 0.5).to(dtype)
    denom = fg_soft.sum(dim=(2, 3)).clamp_min(1e-6)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
    cx = (fg_soft * xx).sum(dim=(2, 3)) / denom
    cy = (fg_soft * yy).sum(dim=(2, 3)) / denom
    sx = torch.sqrt(((fg_soft * (xx - cx[:, :, None, None]) ** 2).sum(dim=(2, 3)) / denom).clamp_min(1e-8))
    sy = torch.sqrt(((fg_soft * (yy - cy[:, :, None, None]) ** 2).sum(dim=(2, 3)) / denom).clamp_min(1e-8))
    edge_sum = edge.sum(dim=(2, 3)).clamp_min(1e-6)
    boundary_entropy = (entropy * edge).sum(dim=(2, 3)) / edge_sum
    scalars = [
        prob.mean(dim=(2, 3)), prob.var(dim=(2, 3), unbiased=False), entropy.mean(dim=(2, 3)), boundary_entropy,
        lowconf.mean(dim=(2, 3)), torch.abs(prob - 0.5).mean(dim=(2, 3)), fg_soft.mean(dim=(2, 3)),
        fg_hard.mean(dim=(2, 3)), (prob > 0.9).to(dtype).mean(dim=(2, 3)), edge.mean(dim=(2, 3)), cx, cy, sx, sy,
    ]
    out = torch.cat(scalars, dim=1).float()
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


class HierarchicalRoutePredictorV3(nn.Module):
    """Hierarchical router: correct? -> add/remove/boundary -> local-FP/tail.

    The hierarchy directly reflects the failure mode seen in MDCR-v2: a flat five-way
    classifier underused rare routes. Factorization gives rare tail cleanup a focused
    binary decision rather than asking it to compete with all other classes at once.
    """

    scalar_dim = 14

    def __init__(self, width: int = 32, hidden: int = 96, dropout: float = 0.15):
        super().__init__()
        self.stem = ConvGNAct(7, width)
        self.blocks = nn.Sequential(
            ResidualDilatedBlock(width, 1), ResidualDilatedBlock(width, 2),
            ResidualDilatedBlock(width, 4), ResidualDilatedBlock(width, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = width + self.scalar_dim
        self.shared = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Dropout(float(dropout)))
        self.correction_head = nn.Linear(hidden, 2)   # no_correction / correction
        self.family_head = nn.Linear(hidden, 3)       # recover_fn / remove_like / boundary_align
        self.remove_head = nn.Linear(hidden, 2)       # remove_fp / tail_cleanup

    def _features(self, image: torch.Tensor, base_logits: torch.Tensor):
        prob, edge, entropy, lowconf = inference_cue_maps(base_logits)
        x = torch.cat([image.float(), prob, edge, entropy, lowconf], dim=1)
        feat = self.blocks(self.stem(x))
        pooled = self.pool(feat).flatten(1)
        cues = inference_scalar_cues(prob, edge, entropy, lowconf)
        h = self.shared(torch.cat([pooled, cues], dim=1))
        return h, cues

    @staticmethod
    def compose_route_probabilities(correction_logits: torch.Tensor, family_logits: torch.Tensor, remove_logits: torch.Tensor) -> torch.Tensor:
        pc = torch.softmax(correction_logits.float(), dim=1)
        pf = torch.softmax(family_logits.float(), dim=1)
        pr = torch.softmax(remove_logits.float(), dim=1)
        p_no = pc[:, 0]
        p_corr = pc[:, 1]
        p_recover = p_corr * pf[:, 0]
        p_remove_like = p_corr * pf[:, 1]
        p_boundary = p_corr * pf[:, 2]
        p_remove_fp = p_remove_like * pr[:, 0]
        p_tail = p_remove_like * pr[:, 1]
        p = torch.stack([p_no, p_recover, p_remove_fp, p_boundary, p_tail], dim=1)
        return (p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)

    def forward_hierarchy(self, image: torch.Tensor, base_logits: torch.Tensor):
        h, cues = self._features(image, base_logits)
        corr = self.correction_head(h)
        fam = self.family_head(h)
        rem = self.remove_head(h)
        probs = self.compose_route_probabilities(corr, fam, rem)
        route_logits = torch.log(probs)
        return route_logits, cues, {"correction": corr, "family": fam, "remove": rem}

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor):
        logits, cues, _ = self.forward_hierarchy(image, base_logits)
        return logits, cues


class ExpertAdapter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(ResidualDilatedBlock(width, 1), ResidualDilatedBlock(width, 2))
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(feat))


class CorrectionExpertsV3(nn.Module):
    """Shared low-level geometry encoder + route-specific adapters/heads."""

    def __init__(self, width: int = 32, residual_scale: float = 2.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = ConvGNAct(7, width)
        self.shared = nn.Sequential(
            ResidualDilatedBlock(width, 1), ResidualDilatedBlock(width, 2), ResidualDilatedBlock(width, 4)
        )
        self.adapters = nn.ModuleList([ExpertAdapter(width) for _ in range(len(ROUTES) - 1)])

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        prob, edge, entropy, lowconf = inference_cue_maps(base_logits)
        x = torch.cat([image.float(), prob, edge, entropy, lowconf], dim=1)
        feat = self.shared(self.stem(x))
        learned = [self.residual_scale * torch.tanh(branch(feat)) for branch in self.adapters]
        zero = torch.zeros_like(learned[0])
        return torch.stack([zero] + learned, dim=1)


class MDCRV3Model(nn.Module):
    """Frozen source + hierarchical deployment router + explicit correction experts."""

    def __init__(self, base: nn.Module, *, router_width: int = 32, router_hidden: int = 96,
                 router_dropout: float = 0.15, expert_width: int = 32, residual_scale: float = 2.0,
                 correction_margin: float = 0.0):
        super().__init__()
        self.base = base
        self.router = HierarchicalRoutePredictorV3(router_width, router_hidden, router_dropout)
        self.experts = CorrectionExpertsV3(expert_width, residual_scale)
        self.correction_margin = float(correction_margin)
        self.freeze_base()

    def freeze_base(self):
        for p in self.base.parameters(): p.requires_grad_(False)
        self.base.eval()

    def freeze_router(self):
        for p in self.router.parameters(): p.requires_grad_(False)
        self.router.eval()

    def unfreeze_router(self):
        for p in self.router.parameters(): p.requires_grad_(True)
        self.router.train()

    def freeze_experts(self):
        for p in self.experts.parameters(): p.requires_grad_(False)
        self.experts.eval()

    def unfreeze_experts(self):
        for p in self.experts.parameters(): p.requires_grad_(True)
        self.experts.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def source_logits(self, image: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        with torch.no_grad():
            return self.base(image)

    def _hard_route_ids(self, route_logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(route_logits.float(), dim=1)
        if self.correction_margin <= 0.0:
            return p.argmax(dim=1)
        no = p[:, 0]
        cp, ci = p[:, 1:].max(dim=1)
        return torch.where((cp - no) >= self.correction_margin, ci + 1, torch.zeros_like(ci))

    def route(self, image: torch.Tensor, base_logits: torch.Tensor | None = None):
        if base_logits is None: base_logits = self.source_logits(image)
        logits, cues = self.router(image, base_logits.detach())
        return logits, torch.softmax(logits.float(), dim=1), cues

    def forward(self, image: torch.Tensor, *, return_parts: bool = False, hard_gate: bool = True,
                forced_route_ids: torch.Tensor | None = None, soft_gate: bool = False):
        base_logits = self.source_logits(image)
        route_logits, route_prob, cues = self.route(image, base_logits)
        expert_residuals = self.experts(image, base_logits.detach())
        if forced_route_ids is not None:
            rid = forced_route_ids.to(image.device).long()
            gate_prob = F.one_hot(rid, num_classes=len(ROUTES)).to(expert_residuals.dtype)
        elif soft_gate and not hard_gate:
            gate_prob = route_prob.to(expert_residuals.dtype)
        else:
            rid = self._hard_route_ids(route_logits)
            gate_prob = F.one_hot(rid, num_classes=len(ROUTES)).to(expert_residuals.dtype)
        residual = (gate_prob[:, :, None, None, None] * expert_residuals).sum(dim=1)
        final_logits = base_logits + residual
        if return_parts:
            return final_logits, base_logits, residual, route_logits, gate_prob, expert_residuals, cues
        return final_logits
