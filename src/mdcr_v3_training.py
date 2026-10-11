from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from .discordance import ROUTES, ROUTE_TO_ID


@dataclass
class RoutePolicyTable:
    by_case: dict[str, dict]
    counts: dict[str, int]


def load_policy_table(path: str, route_column: str) -> RoutePolicyTable:
    df = pd.read_csv(path)
    if route_column not in df.columns:
        raise ValueError(f"route_column={route_column!r} not in {path}; available={list(df.columns)}")
    by_case, labels = {}, []
    for _, r in df.iterrows():
        route = str(r[route_column])
        if route not in ROUTE_TO_ID: route = "no_correction"
        labels.append(route)
        by_case[str(r.case_id)] = {"route": route, "route_id": ROUTE_TO_ID[route], "oracle_gain": float(r.get("oracle_gain", 0.0))}
    return RoutePolicyTable(by_case=by_case, counts=dict(Counter(labels)))


def batch_route_ids(meta, policy: RoutePolicyTable, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(policy.by_case.get(str(cid), {"route_id": 0})["route_id"]) for cid in meta["case_id"]], dtype=torch.long, device=device)


def batch_gain_weights(meta, policy: RoutePolicyTable, device: torch.device, gain_weight: float = 0.0) -> torch.Tensor:
    vals=[]
    for cid in meta["case_id"]:
        g=float(policy.by_case.get(str(cid), {"oracle_gain":0.0}).get("oracle_gain",0.0))
        vals.append(1.0 + float(gain_weight) * min(max(g,0.0),0.20)/0.20)
    return torch.tensor(vals,dtype=torch.float32,device=device)


def balanced_sampler(dataset, policy: RoutePolicyTable, *, seed: int=42, max_ratio: float=20.0, correction_only: bool=False) -> WeightedRandomSampler:
    counts=np.asarray([max(policy.counts.get(r,0),1) for r in ROUTES],dtype=np.float64)
    rw=np.minimum(counts.max()/counts,float(max_ratio))
    weights=[]
    for _,row in dataset.df.iterrows():
        rid=int(policy.by_case.get(str(row.get("case_id","")),{"route_id":0})["route_id"])
        weights.append(0.0 if (correction_only and rid == 0) else float(rw[rid]))
    return WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=len(weights),replacement=True,generator=torch.Generator().manual_seed(int(seed)))


def _weights(counts: list[int], power: float, max_ratio: float, device) -> torch.Tensor:
    a=np.asarray([max(int(v),1) for v in counts],dtype=np.float64)
    w=np.clip((a.max()/a)**float(power),1.0,float(max_ratio)); w=w/w.mean()
    return torch.tensor(w,dtype=torch.float32,device=device)


def hierarchical_class_weights(policy: RoutePolicyTable, power: float, max_ratio: float, device):
    c=policy.counts
    no=c.get("no_correction",0); corr=sum(c.get(r,0) for r in ROUTES[1:])
    recover=c.get("recover_fn",0); remove=c.get("remove_fp",0)+c.get("tail_cleanup",0); boundary=c.get("boundary_align",0)
    local=c.get("remove_fp",0); tail=c.get("tail_cleanup",0)
    return {
        "correction": _weights([no,corr],power,max_ratio,device),
        "family": _weights([recover,remove,boundary],power,max_ratio,device),
        "remove": _weights([local,tail],power,max_ratio,device),
    }


def focal_cross_entropy(logits, targets, *, class_weights=None, gamma: float=1.5, label_smoothing: float=0.0, reduction="mean"):
    ce=F.cross_entropy(logits,targets,weight=class_weights,reduction="none",label_smoothing=float(label_smoothing))
    pt=torch.softmax(logits.float(),dim=1).gather(1,targets[:,None]).squeeze(1).clamp_min(1e-8)
    loss=((1.0-pt)**float(gamma))*ce
    if reduction=="none": return loss
    if reduction=="sum": return loss.sum()
    return loss.mean()


def hierarchical_route_loss(node_logits: dict[str,torch.Tensor], route_ids: torch.Tensor, *, class_weights: dict[str,torch.Tensor], gamma: float, label_smoothing: float, weights: dict):
    corr_t=(route_ids!=0).long()
    l_corr=focal_cross_entropy(node_logits["correction"].float(),corr_t,class_weights=class_weights["correction"],gamma=gamma,label_smoothing=label_smoothing)
    corr_mask=route_ids!=0
    l_family=node_logits["correction"].new_zeros(())
    family_acc=node_logits["correction"].new_tensor(float("nan"))
    if corr_mask.any():
        y=route_ids[corr_mask]
        fam=torch.where(y==ROUTE_TO_ID["recover_fn"],0,torch.where((y==ROUTE_TO_ID["remove_fp"])|(y==ROUTE_TO_ID["tail_cleanup"]),1,2)).long()
        fl=node_logits["family"][corr_mask]
        l_family=focal_cross_entropy(fl.float(),fam,class_weights=class_weights["family"],gamma=gamma,label_smoothing=label_smoothing)
        family_acc=(fl.argmax(1)==fam).float().mean()
    rm_mask=(route_ids==ROUTE_TO_ID["remove_fp"])|(route_ids==ROUTE_TO_ID["tail_cleanup"])
    l_remove=node_logits["correction"].new_zeros(())
    remove_acc=node_logits["correction"].new_tensor(float("nan"))
    if rm_mask.any():
        y=(route_ids[rm_mask]==ROUTE_TO_ID["tail_cleanup"]).long()
        rl=node_logits["remove"][rm_mask]
        l_remove=focal_cross_entropy(rl.float(),y,class_weights=class_weights["remove"],gamma=gamma,label_smoothing=label_smoothing)
        remove_acc=(rl.argmax(1)==y).float().mean()
    total=float(weights.get("correction",1.0))*l_corr+float(weights.get("family",1.0))*l_family+float(weights.get("remove",0.7))*l_remove
    corr_acc=(node_logits["correction"].argmax(1)==corr_t).float().mean()
    return total,{"correction":l_corr,"family":l_family,"remove":l_remove,"correction_acc":corr_acc,"family_acc":family_acc,"remove_acc":remove_acc}


def route_direction_loss(residual: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
    per=residual.new_zeros((residual.shape[0],))
    mean_neg=F.relu(-residual).mean(dim=(1,2,3)); mean_pos=F.relu(residual).mean(dim=(1,2,3))
    per=torch.where(route_ids==ROUTE_TO_ID["recover_fn"],mean_neg,per)
    suppress=(route_ids==ROUTE_TO_ID["remove_fp"])|(route_ids==ROUTE_TO_ID["tail_cleanup"])
    return torch.where(suppress,mean_pos,per)


def _hard_boundary(mask: torch.Tensor) -> torch.Tensor:
    m=mask.float()
    dil=F.max_pool2d(m,3,1,1)
    ero=1.0-F.max_pool2d(1.0-m,3,1,1)
    return (dil-ero).clamp(0,1)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    r=max(int(radius),0)
    if r==0: return mask.float()
    k=2*r+1
    return F.max_pool2d(mask.float(),k,1,r)


def explicit_correction_losses(final_logits: torch.Tensor, source_logits: torch.Tensor, gt: torch.Tensor, route_ids: torch.Tensor, *, threshold: float=0.5, boundary_radius: int=3, tail_radius: int=8, residual_delta: float=1.5):
    """Route-specific supervision on the actual residual error pixels.

    recover_fn: supervise GT-positive/source-negative pixels.
    remove_fp: supervise source-positive/GT-negative pixels.
    boundary_align: supervise source/GT disagreement inside a boundary band.
    tail_cleanup: supervise FP pixels outside a GT neighborhood (fallback to FP if absent).
    Unselected pixels are explicitly preserved toward the frozen source probability.
    """
    src_p=torch.sigmoid(source_logits.float()).detach(); src=(src_p>=float(threshold)).float(); g=(gt.float()>=0.5).float()
    fn=g*(1.0-src); fp=src*(1.0-g); err=(src!=g).float()
    band=_dilate((_hard_boundary(src)+_hard_boundary(g)).clamp_max(1.0),int(boundary_radius))
    boundary_err=err*band
    tail=fp*(1.0-_dilate(g,int(tail_radius)).clamp(0,1))
    has_tail=(tail.sum(dim=(1,2,3),keepdim=True)>0).float(); tail=has_tail*tail+(1.0-has_tail)*fp

    target_mask=torch.zeros_like(g)
    target_mask=torch.where((route_ids==ROUTE_TO_ID["recover_fn"])[:,None,None,None],fn,target_mask)
    target_mask=torch.where((route_ids==ROUTE_TO_ID["remove_fp"])[:,None,None,None],fp,target_mask)
    target_mask=torch.where((route_ids==ROUTE_TO_ID["boundary_align"])[:,None,None,None],boundary_err,target_mask)
    target_mask=torch.where((route_ids==ROUTE_TO_ID["tail_cleanup"])[:,None,None,None],tail,target_mask)

    bce=F.binary_cross_entropy_with_logits(final_logits.float(),g,reduction="none")
    denom=target_mask.sum(dim=(1,2,3)).clamp_min(1.0)
    target_bce=(bce*target_mask).sum(dim=(1,2,3))/denom

    preserve=(1.0-target_mask)
    # Do not use no-correction cases to train experts: their residual is exact zero by construction.
    active=(route_ids!=0).float()
    pden=preserve.sum(dim=(1,2,3)).clamp_min(1.0)
    preserve_l1=(torch.abs(torch.sigmoid(final_logits.float())-src_p)*preserve).sum(dim=(1,2,3))/pden

    desired=torch.zeros_like(final_logits.float())
    desired=desired+float(residual_delta)*fn-float(residual_delta)*fp
    # boundary errors need signed add/remove according to GT-source mismatch.
    desired=torch.where(boundary_err>0, float(residual_delta)*(2.0*g-1.0), desired)
    desired=torch.where(tail>0, torch.full_like(desired,-float(residual_delta)), desired)
    residual=(final_logits.float()-source_logits.float().detach())
    delta=F.smooth_l1_loss(residual,desired,reduction="none")
    delta=(delta*target_mask).sum(dim=(1,2,3))/denom
    return target_bce*active,preserve_l1*active,delta*active,target_mask
