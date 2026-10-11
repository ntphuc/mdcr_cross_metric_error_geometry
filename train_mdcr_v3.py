from __future__ import annotations

import argparse,csv,json,shutil,time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler,autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import BRISC2025Dataset
from src.discordance import DEFAULT_ROUTE_WEIGHTS,ROUTES
from src.embr_losses import geometry_ramp,per_sample_adaptive_tversky,per_sample_bce_dice,per_sample_boundary_dice,per_sample_correctness_anchor,per_sample_multiscale_boundary_distance,per_sample_residual_penalty
from src.mdcr_v3_model import MDCRV3Model
from src.mdcr_v3_training import balanced_sampler,batch_gain_weights,batch_route_ids,explicit_correction_losses,load_policy_table,route_direction_loss
from src.metrics import case_metrics
from src.model_factory import architecture_from_config,build_segmentor
from src.utils import get_device,load_yaml,seed_everything


def finite_mean(x):
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]; return float(a.mean()) if len(a) else float('inf')


def route_loss_parameters(route_ids,device):
    vals={k:[] for k in ['boundary','msbd','tversky','anchor','residual','alpha','beta']}
    for r in route_ids.detach().cpu().tolist():
        w=DEFAULT_ROUTE_WEIGHTS[ROUTES[int(r)]]
        for k,v in [('boundary',w.boundary),('msbd',w.msbd),('tversky',w.tversky),('anchor',w.anchor),('residual',w.residual),('alpha',w.alpha_fp),('beta',w.beta_fn)]: vals[k].append(v)
    return {k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in vals.items()}


def selection_utility(s,cfg):
    hs=float(cfg.get('selection_hd95_scale',16.0)); ass=float(cfg.get('selection_assd_scale',4.0)); w=cfg.get('selection_weights',{}) or {}
    hd=np.exp(-float(s['hd95'])/hs) if np.isfinite(s['hd95']) else 0.0; ad=np.exp(-float(s['assd'])/ass) if np.isfinite(s['assd']) else 0.0
    return float(w.get('dice',1.0)*s['dice']+w.get('bf1',0.15)*s['bf1']+w.get('hd95',0.05)*hd+w.get('assd',0.03)*ad)


def forced_ids(meta,policy,device):
    return batch_route_ids(meta,policy,device) if policy is not None else None


@torch.no_grad()
def exact_validation(model,loader,device,threshold,bf1_tolerance,force_policy=None):
    model.eval(); rows=[]; rc=np.zeros(len(ROUTES),dtype=int); identity=0.0
    for images,masks,meta in loader:
        images=images.to(device); fid=forced_ids(meta,force_policy,device)
        final,base,residual,_,gp,_,_=model(images,return_parts=True,hard_gate=True,forced_route_ids=fid)
        rid=int(gp.argmax(1).item()); rc[rid]+=1
        if rid==0: identity=max(identity,float((final-base).abs().max().cpu()))
        prob=torch.sigmoid(final).cpu().numpy()[0,0]; gt=masks.numpy()[0,0]; rows.append(case_metrics(prob,gt,threshold,bf1_tolerance=bf1_tolerance))
    return {'dice':float(np.mean([r['dice'] for r in rows])),'iou':float(np.mean([r['iou'] for r in rows])),'bf1':float(np.mean([r['bf1'] for r in rows])),'hd95':finite_mean([r['hd95'] for r in rows]),'assd':finite_mean([r['assd'] for r in rows]),'misses':int(sum(r['gt_pixels']>0 and r['pred_pixels']==0 for r in rows)),'predicted_route_counts':{ROUTES[i]:int(rc[i]) for i in range(len(ROUTES))},'no_correction_identity_max_abs_logit_diff':identity}


def payload(model,bcfg,cfg,seed,epoch,source_ckpt,router_ckpt,learned,oracle,expert_route_column):
    return {'model':model.state_dict(),'config':bcfg,'mdcr_v3_config':cfg,'seed':seed,'epoch':epoch,'source_checkpoint':str(source_ckpt),'router_checkpoint':str(router_ckpt),'validation':learned,'oracle_validation':oracle,'expert_route_column':expert_route_column,'method':'MDCR-v3','architecture':architecture_from_config(bcfg),'correction_margin':float(model.correction_margin)}


def main():
    ap=argparse.ArgumentParser(description='MDCR-v3 experts: explicit route-specific correction targets + raw oracle ceiling.')
    ap.add_argument('--config',default='configs/mdcr_v3.yaml'); ap.add_argument('--source-checkpoint',required=True); ap.add_argument('--router-checkpoint',required=True)
    ap.add_argument('--policy-csv',required=True); ap.add_argument('--val-policy-csv',required=True); ap.add_argument('--expert-route-column',default='oracle_route'); ap.add_argument('--oracle-route-column',default='oracle_route')
    ap.add_argument('--manifest',default='data/brisc2025_manifest.csv'); ap.add_argument('--data-root',default='datasets/brisc2025'); ap.add_argument('--device',default='auto'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); cfg=load_yaml(a.config); seed_everything(a.seed); device=get_device(a.device); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    source=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False); bcfg=source['config']; rb=torch.load(a.router_checkpoint,map_location='cpu',weights_only=False)
    train_policy=load_policy_table(a.policy_csv,a.expert_route_column); val_oracle=load_policy_table(a.val_policy_csv,a.oracle_route_column)
    tr=BRISC2025Dataset(a.manifest,'train',a.data_root,int(bcfg['image_size']),True,bcfg.get('imagenet_norm',True)); va=BRISC2025Dataset(a.manifest,'val',a.data_root,int(bcfg['image_size']),False,bcfg.get('imagenet_norm',True))
    sampler=balanced_sampler(tr,train_policy,seed=a.seed+7000,max_ratio=float(cfg.get('expert_sampler_max_ratio',30.0)),correction_only=True)
    nw=int(cfg.get('num_workers',2)); kw=dict(num_workers=nw,pin_memory=device.type=='cuda')
    if nw>0: kw.update(persistent_workers=bool(cfg.get('persistent_workers',True)),prefetch_factor=int(cfg.get('prefetch_factor',2)))
    tl=DataLoader(tr,batch_size=int(cfg.get('expert_batch_size',8)),sampler=sampler,**kw); vl=DataLoader(va,batch_size=1,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    base=build_segmentor(architecture_from_config(bcfg),pretrained_encoder=False); base.load_state_dict(source['model'])
    model=MDCRV3Model(base,router_width=int(cfg.get('router_width',32)),router_hidden=int(cfg.get('router_hidden',96)),router_dropout=float(cfg.get('router_dropout',0.15)),expert_width=int(cfg.get('expert_width',32)),residual_scale=float(cfg.get('residual_scale',2.0)),correction_margin=float(rb.get('correction_margin',0.0))).to(device)
    model.router.load_state_dict(rb['router_state']); model.freeze_base(); model.freeze_router(); model.unfreeze_experts()
    opt=AdamW(model.experts.parameters(),lr=float(cfg.get('expert_lr',3e-4)),weight_decay=float(cfg.get('weight_decay',1e-4))); epochs=int(cfg.get('expert_epochs',18)); sch=CosineAnnealingLR(opt,T_max=max(1,epochs)); expert_amp=bool(cfg.get('expert_amp',False) and device.type=='cuda'); scaler=GradScaler('cuda',enabled=expert_amp)
    threshold=float(cfg.get('threshold',bcfg.get('threshold',0.5))); bf1=int(cfg.get('bf1_tolerance',2))

    source_learned=exact_validation(model,vl,device,threshold,bf1); source_oracle=exact_validation(model,vl,device,threshold,bf1,val_oracle); source_learned['utility']=selection_utility(source_learned,cfg); source_oracle['utility']=selection_utility(source_oracle,cfg)
    p0=payload(model,bcfg,cfg,a.seed,0,a.source_checkpoint,a.router_checkpoint,source_learned,source_oracle,a.expert_route_column); torch.save(p0,out/'best_utility.pt'); torch.save(p0,out/'best_dice.pt')
    best_u=source_learned['utility']; best_d=source_learned['dice']; best_epoch=0
    # Raw expert ceiling deliberately excludes epoch 0/fallback. This fixes the v2 oracle bug.
    best_oracle_u=-float('inf'); best_oracle_d=-float('inf'); best_oracle_u_epoch=-1; best_oracle_d_epoch=-1; best_raw_learned_u=-float('inf')
    (out/'source_validation.json').write_text(json.dumps({'learned':source_learned,'oracle':source_oracle},indent=2),encoding='utf-8')
    fields=['epoch','train_total','train_explicit','train_preserve','train_delta','train_base','train_boundary','train_msbd','train_tversky','train_anchor','train_direction','fp32_retries','geometry_fallback_batches','max_grad_norm','learned_dice','learned_bf1','learned_hd95','learned_assd','learned_utility','oracle_dice','oracle_bf1','oracle_hd95','oracle_assd','oracle_utility','eligible','seconds']
    with open(out/'history.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ep in range(1,epochs+1):
            t0=time.time(); model.train(); model.base.eval(); model.router.eval(); model.experts.train(); ramp=geometry_ramp(ep,int(cfg.get('geometry_warmup_epochs',1)),int(cfg.get('geometry_ramp_epochs',3)))
            acc={k:[] for k in ['total','explicit','preserve','delta','base','boundary','msbd','tversky','anchor','direction']}
            fp32_retries=0; geometry_fallback_batches=0; max_grad_norm=0.0

            def compute_batch_loss(images,masks,rid,cw,rp, *, use_amp=False, use_geometry=True):
                # Experts are intentionally trained in FP32 by default. The geometric
                # objectives contain normalized boundary ratios whose backward pass is
                # more sensitive than the forward scalar loss. AMP can therefore be
                # enabled explicitly, but is not the publication default.
                with autocast(device_type=device.type,enabled=bool(use_amp)):
                    final,base_logits,residual,_,_,_,_=model(images,return_parts=True,forced_route_ids=rid)
                fl,bl,mk,rl=final.float(),base_logits.float(),masks.float(),residual.float()
                base_l=per_sample_bce_dice(fl,mk)
                # Do not even build the geometric backward graph while its epoch ramp is
                # zero. 0 * an ill-conditioned derivative can still produce NaN.
                if use_geometry and ramp>0.0:
                    bnd=per_sample_boundary_dice(fl,mk)
                    msbd=per_sample_multiscale_boundary_distance(fl,mk)
                    tv=per_sample_adaptive_tversky(fl,mk,rp['alpha'],rp['beta'])
                else:
                    z=torch.zeros_like(base_l); bnd=z; msbd=z; tv=z
                anchor=per_sample_correctness_anchor(fl,bl,mk,boundary_radius=int(cfg.get('anchor_boundary_radius',3)))
                direction=route_direction_loss(rl,rid); residual_l=per_sample_residual_penalty(rl)
                expl,preserve,delta,_=explicit_correction_losses(fl,bl,mk,rid,threshold=threshold,boundary_radius=int(cfg.get('explicit_boundary_radius',4)),tail_radius=int(cfg.get('explicit_tail_radius',10)),residual_delta=float(cfg.get('explicit_residual_delta',1.5)))
                geom=(rp['boundary']*bnd+rp['msbd']*msbd+rp['tversky']*tv) if use_geometry else torch.zeros_like(base_l)
                per=(float(cfg.get('explicit_target_weight',1.4))*expl+float(cfg.get('explicit_preserve_weight',0.35))*preserve+float(cfg.get('explicit_delta_weight',0.15))*delta+float(cfg.get('base_weight',0.30))*base_l+float(cfg.get('geometry_scale',0.60))*ramp*geom+float(cfg.get('anchor_scale',0.35))*rp['anchor']*anchor+float(cfg.get('residual_penalty_scale',0.50))*rp['residual']*residual_l+float(cfg.get('route_direction_weight',0.08))*direction)*cw
                loss=per.mean()
                parts={'explicit':expl.mean(),'preserve':preserve.mean(),'delta':delta.mean(),'base':base_l.mean(),'boundary':bnd.mean(),'msbd':msbd.mean(),'tversky':tv.mean(),'anchor':anchor.mean(),'direction':direction.mean()}
                return loss,parts

            def gradients_are_finite():
                for p in model.experts.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        return False
                return True

            for images,masks,meta in tl:
                images=images.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True); rid=batch_route_ids(meta,train_policy,device); cw=batch_gain_weights(meta,train_policy,device,float(cfg.get('oracle_gain_weight',0.5))); rp=route_loss_parameters(rid,device)
                # First attempt: configured precision, full objective.
                used_fp32_retry=False; used_geometry_fallback=False
                opt.zero_grad(set_to_none=True)
                loss,parts=compute_batch_loss(images,masks,rid,cw,rp,use_amp=expert_amp,use_geometry=True)
                if not torch.isfinite(loss):
                    finite_forward=False
                else:
                    finite_forward=True
                    if expert_amp:
                        scaler.scale(loss).backward(); scaler.unscale_(opt)
                    else:
                        loss.backward()
                finite_backward=finite_forward and gradients_are_finite()

                # If an explicitly enabled AMP run overflows, retry the exact same batch
                # once in FP32 before changing the objective. No per-batch print: retries
                # are summarized in the single epoch log line.
                if not finite_backward and expert_amp:
                    fp32_retries+=1; used_fp32_retry=True; opt.zero_grad(set_to_none=True)
                    loss,parts=compute_batch_loss(images,masks,rid,cw,rp,use_amp=False,use_geometry=True)
                    if torch.isfinite(loss): loss.backward()
                    finite_backward=torch.isfinite(loss) and gradients_are_finite()

                # Last-resort numerical guard: keep the scientifically central explicit
                # correction/preservation objectives for this batch, but omit geometric
                # surrogates if their backward graph is non-finite. This is counted and
                # persisted, never silently hidden.
                if not finite_backward and bool(cfg.get('expert_geometry_fallback',True)):
                    geometry_fallback_batches+=1; used_geometry_fallback=True; opt.zero_grad(set_to_none=True)
                    loss,parts=compute_batch_loss(images,masks,rid,cw,rp,use_amp=False,use_geometry=False)
                    if torch.isfinite(loss): loss.backward()
                    finite_backward=torch.isfinite(loss) and gradients_are_finite()

                if not finite_backward:
                    debug={'epoch':ep,'case_ids':[str(x) for x in meta.get('case_id',[])],'loss':float(loss.detach().cpu()) if torch.isfinite(loss) else None,'expert_amp':expert_amp,'geometry_fallback_attempted':bool(cfg.get('expert_geometry_fallback',True))}
                    (out/'nonfinite_expert_debug.json').write_text(json.dumps(debug,indent=2),encoding='utf-8')
                    raise FloatingPointError(f'non-finite v3 expert gradients epoch={ep}; see {out/"nonfinite_expert_debug.json"}')

                grad_norm=torch.nn.utils.clip_grad_norm_(model.experts.parameters(),float(cfg.get('grad_clip',5.0)),error_if_nonfinite=True)
                max_grad_norm=max(max_grad_norm,float(grad_norm.detach().cpu()))
                if expert_amp and not used_fp32_retry and not used_geometry_fallback:
                    scaler.step(opt); scaler.update()
                else:
                    # A successful FP32 retry/fallback must not be skipped because the
                    # scaler remembers the earlier AMP overflow for this optimizer step.
                    opt.step()
                    if expert_amp:
                        scaler.update(new_scale=max(float(scaler.get_scale())*0.5,1.0))
                for k,v in [('total',loss),*parts.items()]: acc[k].append(float(v.detach().cpu()))
            learned=exact_validation(model,vl,device,threshold,bf1); oracle=exact_validation(model,vl,device,threshold,bf1,val_oracle); lu=selection_utility(learned,cfg); ou=selection_utility(oracle,cfg); learned['utility']=lu; oracle['utility']=ou
            eligible=learned['dice']>=source_learned['dice']-float(cfg.get('selection_dice_margin',0.0010)) and learned['misses']<=source_learned['misses']+int(cfg.get('selection_max_extra_misses',0)); p=payload(model,bcfg,cfg,a.seed,ep,a.source_checkpoint,a.router_checkpoint,learned,oracle,a.expert_route_column); torch.save(p,out/'last_raw.pt')
            if learned['dice']>best_d: best_d=learned['dice']; torch.save(p,out/'best_dice.pt')
            if eligible and lu>best_u+float(cfg.get('selection_min_utility_gain',1e-5)): best_u=lu; best_epoch=ep; torch.save(p,out/'best_utility.pt')
            if lu>best_raw_learned_u: best_raw_learned_u=lu; torch.save(p,out/'best_learned_raw.pt')
            if ou>best_oracle_u: best_oracle_u=ou; best_oracle_u_epoch=ep; torch.save(p,out/'best_oracle_utility.pt')
            if oracle['dice']>best_oracle_d: best_oracle_d=oracle['dice']; best_oracle_d_epoch=ep; torch.save(p,out/'best_oracle_dice.pt')
            row={'epoch':ep,**{f'train_{k}':float(np.mean(v)) if v else 0.0 for k,v in acc.items()},'fp32_retries':fp32_retries,'geometry_fallback_batches':geometry_fallback_batches,'max_grad_norm':max_grad_norm,'learned_dice':learned['dice'],'learned_bf1':learned['bf1'],'learned_hd95':learned['hd95'],'learned_assd':learned['assd'],'learned_utility':lu,'oracle_dice':oracle['dice'],'oracle_bf1':oracle['bf1'],'oracle_hd95':oracle['hd95'],'oracle_assd':oracle['assd'],'oracle_utility':ou,'eligible':int(eligible),'seconds':time.time()-t0}; w.writerow(row); f.flush(); sch.step()
            print(f"experts-v3 epoch={ep:02d}/{epochs} loss={row['train_total']:.4f} learned={learned['dice']:.5f}/{learned['hd95']:.3f} oracle={oracle['dice']:.5f}/{oracle['hd95']:.3f} grad={max_grad_norm:.3f} retry={fp32_retries} geomfb={geometry_fallback_batches} best_oracle_ep={best_oracle_u_epoch} sec={row['seconds']:.1f}",flush=True)
    shutil.copy2(out/'best_utility.pt',out/'deployment.pt')
    decision={'method':'MDCR-v3','source_validation':source_learned,'best_deployment_utility':best_u,'best_deployment_epoch':best_epoch,'safe_fallback_to_source':bool(best_epoch==0),'raw_best_learned_utility':best_raw_learned_u,'raw_oracle_best_utility':best_oracle_u,'raw_oracle_best_utility_epoch':best_oracle_u_epoch,'raw_oracle_best_dice':best_oracle_d,'raw_oracle_best_dice_epoch':best_oracle_d_epoch,'oracle_ceiling_excludes_epoch0':True,'source_frozen':True,'router_frozen':True,'explicit_correction_targets':True}
    (out/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8'); print(f"experts-v3 done deployment_epoch={best_epoch} oracle_ceiling_epoch={best_oracle_u_epoch} fallback={int(best_epoch==0)}",flush=True)

if __name__=='__main__': main()
