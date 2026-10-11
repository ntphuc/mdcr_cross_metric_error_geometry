from __future__ import annotations

import argparse,csv,json,time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from torch.amp import GradScaler,autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import BRISC2025Dataset
from src.discordance import ROUTES
from src.mdcr_v3_model import MDCRV3Model
from src.mdcr_v3_training import balanced_sampler,batch_route_ids,hierarchical_class_weights,hierarchical_route_loss,load_policy_table
from src.model_factory import architecture_from_config,build_segmentor
from src.utils import get_device,load_yaml,seed_everything


def route_metrics(y_true,y_pred):
    labels=list(range(len(ROUTES)))
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),"macro_f1":float(f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
            "per_route_precision":{ROUTES[i]:float(p[i]) for i in labels},"per_route_recall":{ROUTES[i]:float(r[i]) for i in labels},"per_route_f1":{ROUTES[i]:float(f[i]) for i in labels},"per_route_support":{ROUTES[i]:int(s[i]) for i in labels},"confusion":confusion_matrix(y_true,y_pred,labels=labels).tolist()}


def predict_ids(logits,margin):
    x=logits-logits.max(axis=1,keepdims=True); p=np.exp(x); p/=p.sum(axis=1,keepdims=True)
    no=p[:,0]; ci=p[:,1:].argmax(1)+1; cp=p[np.arange(len(p)),ci]
    return np.where((cp-no)>=float(margin),ci,0).astype(np.int64)


def calibrate_margin(logits,y_true,penalty):
    best=None
    for m in np.linspace(-0.30,0.30,61):
        pred=predict_ids(logits,m); met=route_metrics(y_true,pred); mask=y_true==0
        false_corr=float(np.mean(pred[mask]!=0)) if np.any(mask) else 0.0
        obj=met['macro_f1']-float(penalty)*false_corr
        row={"margin":float(m),"objective":float(obj),"false_correction_rate_on_no_correction":false_corr,**{k:met[k] for k in ['accuracy','balanced_accuracy','macro_f1']}}
        if best is None or row['objective']>best[1]['objective']: best=(float(m),row)
    return best


@torch.no_grad()
def validate(model,loader,policy,device):
    model.eval(); ys=[]; lgs=[]
    for images,_,meta in loader:
        images=images.to(device,non_blocking=True); y=batch_route_ids(meta,policy,device); base=model.source_logits(images)
        lg,_,_=model.router.forward_hierarchy(images,base)
        if not torch.isfinite(lg.float()).all(): raise FloatingPointError('non-finite v3 router logits during validation')
        ys.append(y.cpu().numpy()); lgs.append(lg.float().cpu().numpy())
    return np.concatenate(ys),np.concatenate(lgs)


def main():
    ap=argparse.ArgumentParser(description='MDCR-v3 hierarchical deployment router.')
    ap.add_argument('--config',default='configs/mdcr_v3.yaml'); ap.add_argument('--source-checkpoint',required=True)
    ap.add_argument('--train-policy-csv',required=True); ap.add_argument('--val-policy-csv',required=True)
    ap.add_argument('--route-column',default='oracle_route'); ap.add_argument('--val-route-column',default='oracle_route')
    ap.add_argument('--manifest',default='data/brisc2025_manifest.csv'); ap.add_argument('--data-root',default='datasets/brisc2025')
    ap.add_argument('--device',default='auto'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); cfg=load_yaml(a.config); seed_everything(a.seed); device=get_device(a.device); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    source=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False); bcfg=source['config']
    trp=load_policy_table(a.train_policy_csv,a.route_column); vap=load_policy_table(a.val_policy_csv,a.val_route_column)
    tr=BRISC2025Dataset(a.manifest,'train',a.data_root,int(bcfg['image_size']),False,bcfg.get('imagenet_norm',True)); va=BRISC2025Dataset(a.manifest,'val',a.data_root,int(bcfg['image_size']),False,bcfg.get('imagenet_norm',True))
    sampler=balanced_sampler(tr,trp,seed=a.seed,max_ratio=float(cfg.get('route_sampler_max_ratio',20.0)))
    nw=int(cfg.get('num_workers',2)); kw=dict(num_workers=nw,pin_memory=device.type=='cuda')
    if nw>0: kw.update(persistent_workers=bool(cfg.get('persistent_workers',True)),prefetch_factor=int(cfg.get('prefetch_factor',2)))
    tl=DataLoader(tr,batch_size=int(cfg.get('router_batch_size',16)),sampler=sampler,**kw); vl=DataLoader(va,batch_size=int(cfg.get('router_batch_size',16)),shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    base=build_segmentor(architecture_from_config(bcfg),pretrained_encoder=False); base.load_state_dict(source['model'])
    model=MDCRV3Model(base,router_width=int(cfg.get('router_width',32)),router_hidden=int(cfg.get('router_hidden',96)),router_dropout=float(cfg.get('router_dropout',0.15)),expert_width=int(cfg.get('expert_width',32)),residual_scale=float(cfg.get('residual_scale',2.0))).to(device)
    model.freeze_base(); model.freeze_experts(); model.unfreeze_router()
    cw=hierarchical_class_weights(trp,float(cfg.get('route_class_weight_power',0.35)),float(cfg.get('route_class_weight_max',3.0)),device)
    opt=AdamW(model.router.parameters(),lr=float(cfg.get('router_lr',3e-4)),weight_decay=float(cfg.get('weight_decay',1e-4))); epochs=int(cfg.get('router_epochs',24)); sch=CosineAnnealingLR(opt,T_max=max(1,epochs))
    amp=bool(cfg.get('router_amp',False) and device.type=='cuda'); scaler=GradScaler('cuda',enabled=amp); gamma=float(cfg.get('route_focal_gamma',1.5)); smooth=float(cfg.get('route_label_smoothing',0.01)); penalty=float(cfg.get('router_false_correction_penalty',0.20)); hw=cfg.get('hierarchy_loss_weights',{}) or {}
    fields=['epoch','loss','train_acc','corr_loss','family_loss','remove_loss','corr_acc','family_acc','remove_acc','val_accuracy','val_balanced_accuracy','val_macro_f1','margin','safe_objective','seconds']; best=-1e9; best_epoch=0
    with open(out/'history.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ep in range(1,epochs+1):
            t0=time.time(); model.train(); model.base.eval(); model.experts.eval(); model.router.train(); agg={k:[] for k in ['loss','acc','corr','fam','rem','cacc','facc','racc']}
            for images,_,meta in tl:
                images=images.to(device,non_blocking=True); y=batch_route_ids(meta,trp,device); opt.zero_grad(set_to_none=True)
                with autocast(device_type=device.type,enabled=amp):
                    b=model.source_logits(images); lg,_,nodes=model.router.forward_hierarchy(images,b)
                loss,parts=hierarchical_route_loss(nodes,y,class_weights=cw,gamma=gamma,label_smoothing=smooth,weights=hw)
                if not torch.isfinite(loss): raise FloatingPointError(f'non-finite hierarchical router loss epoch={ep}')
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.router.parameters(),float(cfg.get('grad_clip',5.0)),error_if_nonfinite=True); scaler.step(opt); scaler.update()
                agg['loss'].append(float(loss.detach().cpu())); agg['acc'].append(float((lg.argmax(1)==y).float().mean().cpu())); agg['corr'].append(float(parts['correction'].detach().cpu())); agg['fam'].append(float(parts['family'].detach().cpu())); agg['rem'].append(float(parts['remove'].detach().cpu()))
                for dst,key in [('cacc','correction_acc'),('facc','family_acc'),('racc','remove_acc')]:
                    v=parts[key];
                    if torch.isfinite(v): agg[dst].append(float(v.detach().cpu()))
            yv,lv=validate(model,vl,vap,device); margin,safe=calibrate_margin(lv,yv,penalty); pred=predict_ids(lv,margin); vm=route_metrics(yv,pred)
            row={'epoch':ep,'loss':float(np.mean(agg['loss'])),'train_acc':float(np.mean(agg['acc'])),'corr_loss':float(np.mean(agg['corr'])),'family_loss':float(np.mean(agg['fam'])),'remove_loss':float(np.mean(agg['rem'])),'corr_acc':float(np.mean(agg['cacc'])) if agg['cacc'] else np.nan,'family_acc':float(np.mean(agg['facc'])) if agg['facc'] else np.nan,'remove_acc':float(np.mean(agg['racc'])) if agg['racc'] else np.nan,'val_accuracy':vm['accuracy'],'val_balanced_accuracy':vm['balanced_accuracy'],'val_macro_f1':vm['macro_f1'],'margin':margin,'safe_objective':safe['objective'],'seconds':time.time()-t0}
            w.writerow(row); f.flush(); sch.step()
            payload={'method':'MDCR-v3-hierarchical-router','router_schema':'hierarchical_v3','epoch':ep,'seed':a.seed,'architecture':architecture_from_config(bcfg),'config':bcfg,'mdcr_v3_config':cfg,'source_checkpoint':str(a.source_checkpoint),'route_column':a.route_column,'val_route_column':a.val_route_column,'router_state':model.router.state_dict(),'correction_margin':float(margin),'validation':vm,'safe_objective':float(safe['objective']),'class_counts':trp.counts}
            torch.save(payload,out/'last.pt')
            if safe['objective']>best: best=float(safe['objective']); best_epoch=ep; torch.save(payload,out/'best.pt')
            print(f"router-v3 epoch={ep:02d}/{epochs} loss={row['loss']:.4f} acc={row['train_acc']:.3f} val_acc={vm['accuracy']:.3f} val_bal={vm['balanced_accuracy']:.3f} val_f1={vm['macro_f1']:.3f} nodes={row['corr_acc']:.2f}/{row['family_acc']:.2f}/{row['remove_acc']:.2f} margin={margin:.3f} sec={row['seconds']:.1f}",flush=True)
    bestp=torch.load(out/'best.pt',map_location='cpu',weights_only=False); meta={'best_epoch':best_epoch,'best_safe_objective':best,'best_validation':bestp['validation'],'correction_margin':bestp['correction_margin'],'train_counts':trp.counts,'route_column':a.route_column,'router_schema':'hierarchical_v3'}; (out/'router_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(f"router-v3 done best_epoch={best_epoch} macro_f1={meta['best_validation']['macro_f1']:.4f}",flush=True)

if __name__=='__main__': main()
