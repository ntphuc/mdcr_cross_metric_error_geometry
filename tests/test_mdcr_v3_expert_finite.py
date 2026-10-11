import torch

from src.embr_losses import per_sample_multiscale_boundary_distance
from src.mdcr_v3_training import explicit_correction_losses, route_direction_loss


def test_msbd_backward_finite_for_saturated_logits():
    for value in (-100.0, 100.0):
        logits=torch.full((2,1,32,32),value,requires_grad=True)
        gt=torch.zeros((2,1,32,32)); gt[:, :, 8:24, 8:24]=1.0
        loss=per_sample_multiscale_boundary_distance(logits,gt).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()


def test_explicit_losses_backward_finite_all_routes():
    base=torch.randn(4,1,32,32)*8.0
    residual=torch.zeros_like(base,requires_grad=True)
    final=base+residual
    gt=(torch.rand_like(base)>0.7).float()
    rid=torch.tensor([1,2,3,4],dtype=torch.long)
    expl,preserve,delta,_=explicit_correction_losses(final,base,gt,rid)
    direction=route_direction_loss(residual,rid)
    loss=(expl+preserve+delta+direction).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(residual.grad).all()
