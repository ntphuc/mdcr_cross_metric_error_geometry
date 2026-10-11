from __future__ import annotations

import argparse,json,os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.discordance import ROUTES,ROUTE_TO_ID
from src.mdcr_v3_model import MDCRV3Model
from src.metrics import case_metrics
from src.model_factory import architecture_from_config,build_segmentor
from src.utils import get_device,save_json


def first(v):
    if isinstance(v,(list,tuple)): return v[0]
    if torch.is_tensor(v): return v.item() if v.numel()==1 else v[0].item()
    return v

def finite_mean(v):
    x=np.asarray(v,dtype=float); x=x[np.isfinite(x)]; return float(x.mean()) if len(x) else None

def load_force(path,col):
    if not path: return None
    d=pd.read_csv(path)
    if not col or col not in d.columns: raise ValueError(f'force route column {col!r} missing from {path}')
    return {str(r.case_id):ROUTE_TO_ID.get(str(r[col]),0) for _,r in d.iterrows()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',default='data/brisc2025_manifest.csv'); ap.add_argument('--data-root',default='datasets/brisc2025'); ap.add_argument('--split',default='val',choices=['train','val','test']); ap.add_argument('--device',default='auto'); ap.add_argument('--output-dir',required=True); ap.add_argument('--force-policy-csv',default=None); ap.add_argument('--force-route-column',default=None)
    a=ap.parse_args(); device=get_device(a.device); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); bcfg=ck['config']; cfg=ck.get('mdcr_v3_config',{}) or {}; base=build_segmentor(architecture_from_config(bcfg),pretrained_encoder=False)
    model=MDCRV3Model(base,router_width=int(cfg.get('router_width',32)),router_hidden=int(cfg.get('router_hidden',96)),router_dropout=float(cfg.get('router_dropout',0.15)),expert_width=int(cfg.get('expert_width',32)),residual_scale=float(cfg.get('residual_scale',2.0)),correction_margin=float(ck.get('correction_margin',0.0))); model.load_state_dict(ck['model']); model.to(device).eval()
    force=load_force(a.force_policy_csv,a.force_route_column); ds=BRISC2025Dataset(a.manifest,a.split,a.data_root,int(bcfg['image_size']),False,bcfg.get('imagenet_norm',True)); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0); th=float(cfg.get('threshold',bcfg.get('threshold',0.5))); bf=int(cfg.get('bf1_tolerance',2)); rows=[]; identity=0.0
    with torch.no_grad():
        for images,masks,meta in tqdm(loader,desc=f'mdcr-v3-{a.split}',disable=os.environ.get('TQDM_DISABLE')=='1'):
            cid=str(first(meta['case_id'])); forced=None if force is None else torch.tensor([int(force.get(cid,0))],device=device)
            final,base_logits,residual,route_logits,gp,_,_=model(images.to(device),return_parts=True,forced_route_ids=forced); prob=torch.sigmoid(final).cpu().numpy()[0,0]; gt=masks.numpy()[0,0]; met=case_metrics(prob,gt,th,bf1_tolerance=bf); rid=int(gp.argmax(1).item()); rp=torch.softmax(route_logits.float(),dim=1).cpu().numpy()[0]
            if rid==0: identity=max(identity,float((final-base_logits).abs().max().cpu()))
            rows.append({'case_id':cid,'plane_code':str(first(meta['plane_code'])),'tumor_code':str(first(meta['tumor_code'])),**met,'predicted_route':ROUTES[rid],'route_confidence':float(gp[0,rid].cpu()),'router_argmax':ROUTES[int(np.argmax(rp))],**{f'p_{ROUTES[i]}':float(rp[i]) for i in range(len(ROUTES))},'mean_abs_residual_logit':float(residual.abs().mean().cpu()),'forced_route':ROUTES[rid] if force is not None else ''})
    df=pd.DataFrame(rows); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'per_case_metrics.csv',index=False)
    s={'checkpoint':str(a.checkpoint),'split':a.split,'n_cases':len(df),'architecture':architecture_from_config(bcfg),'routing_mode':'forced' if force is not None else 'learned','force_route_column':a.force_route_column,'dice_mean':float(df.dice.mean()),'iou_mean':float(df.iou.mean()),'bf1_mean':float(df.bf1.mean()),'hd95_mean_finite':finite_mean(df.hd95),'assd_mean_finite':finite_mean(df.assd),'missed_lesion_cases':int(((df.gt_pixels>0)&(df.pred_pixels==0)).sum()),'route_counts':df.predicted_route.value_counts().to_dict(),'mean_route_confidence':float(df.route_confidence.mean()),'no_correction_identity_max_abs_logit_diff':identity,'note':'Forced oracle routing is diagnostic only; official test always uses learned routing.'}
    save_json(s,out/'summary.json'); print(json.dumps(s,indent=2))
if __name__=='__main__': main()
