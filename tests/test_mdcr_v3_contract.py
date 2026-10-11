import torch
import torch.nn as nn
from src.mdcr_v3_model import MDCRV3Model,HierarchicalRoutePredictorV3
from src.mdcr_v3_training import explicit_correction_losses

class TinyBase(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(3,1,3,padding=1)
    def forward(self,x): return self.conv(x)

def test_no_correction_exact_identity():
    torch.manual_seed(0); m=MDCRV3Model(TinyBase(),router_width=8,router_hidden=16,expert_width=8,residual_scale=1.0)
    x=torch.randn(2,3,32,32); r=torch.zeros(2,dtype=torch.long); out=m(x,return_parts=True,forced_route_ids=r)
    assert torch.equal(out[0],out[1]); assert torch.count_nonzero(out[2])==0; assert all(not p.requires_grad for p in m.base.parameters())

def test_hierarchical_route_probs_normalize_and_are_finite():
    torch.manual_seed(1); r=HierarchicalRoutePredictorV3(width=8,hidden=16); x=torch.randn(4,3,32,32); z=torch.randn(4,1,32,32)
    lg,_,nodes=r.forward_hierarchy(x,z); p=torch.softmax(lg,dim=1)
    assert lg.shape==(4,5); assert torch.isfinite(lg).all(); assert torch.allclose(p.sum(1),torch.ones(4),atol=1e-5)
    assert nodes['correction'].shape==(4,2) and nodes['family'].shape==(4,3) and nodes['remove'].shape==(4,2)

def test_explicit_targets_have_finite_losses():
    src=torch.full((4,1,16,16),-2.0); gt=torch.zeros_like(src); gt[0,:,4:8,4:8]=1; src[1,:,4:8,4:8]=2; gt[2,:,5:10,5:10]=1; src[2,:,4:9,4:9]=2; src[3,:,1:3,1:3]=2
    final=src.clone().requires_grad_(True); routes=torch.tensor([1,2,3,4]); a,b,c,mask=explicit_correction_losses(final,src,gt,routes)
    for t in (a,b,c,mask): assert torch.isfinite(t).all()
    (a.mean()+b.mean()+c.mean()).backward(); assert final.grad is not None and torch.isfinite(final.grad).all()
