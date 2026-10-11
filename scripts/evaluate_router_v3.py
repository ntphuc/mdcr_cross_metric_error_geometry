from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import BRISC2025Dataset
from src.discordance import ROUTES,ROUTE_TO_ID
from src.mdcr_v3_model import MDCRV3Model
from src.model_factory import architecture_from_config,build_segmentor
from src.utils import get_device

def metrics(y,p):
    labels=list(range(len(ROUTES))); pr,re,f,s=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),'per_route_recall':{ROUTES[i]:float(re[i]) for i in labels},'per_route_f1':{ROUTES[i]:float(f[i]) for i in labels},'confusion':confusion_matrix(y,p,labels=labels).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--router-checkpoint',required=True); ap.add_argument('--source-checkpoint',required=True); ap.add_argument('--policy-csv',required=True); ap.add_argument('--label-columns',nargs='+',required=True); ap.add_argument('--manifest',default='data/brisc2025_manifest.csv'); ap.add_argument('--data-root',default='datasets/brisc2025'); ap.add_argument('--split',default='val'); ap.add_argument('--device',default='auto'); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    device=get_device(a.device); rb=torch.load(a.router_checkpoint,map_location='cpu',weights_only=False); src=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False); bcfg=src['config']; cfg=rb.get('mdcr_v3_config',{}) or {}; base=build_segmentor(architecture_from_config(bcfg),pretrained_encoder=False); base.load_state_dict(src['model']); model=MDCRV3Model(base,router_width=int(cfg.get('router_width',32)),router_hidden=int(cfg.get('router_hidden',96)),router_dropout=float(cfg.get('router_dropout',0.15)),expert_width=int(cfg.get('expert_width',32)),residual_scale=float(cfg.get('residual_scale',2.0)),correction_margin=float(rb.get('correction_margin',0.0))); model.router.load_state_dict(rb['router_state']); model.to(device).eval()
    ds=BRISC2025Dataset(a.manifest,a.split,a.data_root,int(bcfg['image_size']),False,bcfg.get('imagenet_norm',True)); loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0); pred={}
    with torch.no_grad():
        for images,_,meta in tqdm(loader,desc='router-v3-eval',disable=os.environ.get('TQDM_DISABLE')=='1'):
            images=images.to(device); lg,_,_=model.route(images,model.source_logits(images)); rid=model._hard_route_ids(lg).cpu().tolist()
            for cid,r in zip([str(x) for x in meta['case_id']],rid): pred[cid]=int(r)
    pol=pd.read_csv(a.policy_csv); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for col in a.label_columns:
        if col not in pol.columns: continue
        sub=pol[pol.case_id.astype(str).isin(pred)].copy(); y=np.array([ROUTE_TO_ID.get(str(x),0) for x in sub[col]],dtype=int); p=np.array([pred[str(x)] for x in sub.case_id.astype(str)],dtype=int); m=metrics(y,p); rows.append({'label_column':col,**{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1']}}); pd.DataFrame(m['confusion'],index=ROUTES,columns=ROUTES).to_csv(out/f'confusion_{col}.csv'); (out/f'metrics_{col}.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
