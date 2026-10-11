from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.discordance import metric_profile_features, scalar_quality_from_features
from src.metrics import case_metrics


def exact_n_near_fp(gt: np.ndarray, n: int) -> np.ndarray:
    gt=gt.astype(bool); ring=np.logical_and(binary_dilation(gt,iterations=8),~gt)
    coords=np.argwhere(ring)
    if len(coords)==0: return gt.copy()
    # Nearest background pixels to the object form a shallow local expansion.
    d=distance_transform_edt(~gt); order=np.argsort(d[ring]); chosen=coords[order[:min(n,len(coords))]]
    out=gt.copy(); out[chosen[:,0],chosen[:,1]]=True; return out


def exact_n_distant_fp(gt: np.ndarray, n: int) -> np.ndarray:
    gt=gt.astype(bool); h,w=gt.shape; yy,xx=np.where(gt)
    cy=float(yy.mean()) if len(yy) else h/2; cx=float(xx.mean()) if len(xx) else w/2
    corners=[(4,4),(4,w-5),(h-5,4),(h-5,w-5)]; ty,tx=max(corners,key=lambda p:(p[0]-cy)**2+(p[1]-cx)**2)
    y,x=np.ogrid[:h,:w]; dist=(y-ty)**2+(x-tx)**2
    candidates=np.argwhere(~gt)
    score=dist[~gt]
    # For a connected-ish island, use background pixels closest to a far corner center.
    chosen=candidates[np.argsort(score)[:min(n,len(candidates))]]
    out=gt.copy(); out[chosen[:,0],chosen[:,1]]=True; return out


def main():
    ap=argparse.ArgumentParser(description='Matched-FP-volume geometry benchmark: same FP amount, different geometry.')
    ap.add_argument('--manifest',default='data/brisc2025_manifest.csv'); ap.add_argument('--data-root',default='datasets/brisc2025')
    ap.add_argument('--split',required=True,choices=['train','val','test']); ap.add_argument('--output',required=True); ap.add_argument('--image-size',type=int,default=256)
    ap.add_argument('--max-cases',type=int,default=0); ap.add_argument('--fp-fraction',type=float,default=0.08); ap.add_argument('--min-fp',type=int,default=32); ap.add_argument('--max-fp',type=int,default=256)
    args=ap.parse_args(); ds=BRISC2025Dataset(args.manifest,args.split,args.data_root,args.image_size,False,False); indices=range(len(ds)) if not args.max_cases else range(min(len(ds),args.max_cases))
    rows=[]
    for idx in tqdm(indices,desc=f'matched-geometry-{args.split}', disable=os.environ.get('TQDM_DISABLE') == '1'):
        _,mt,meta=ds[idx]; gt=mt.numpy()[0]>=0.5; n=int(np.clip(round(gt.sum()*args.fp_fraction),args.min_fp,args.max_fp))
        for label,pred in [('remove_fp',exact_n_near_fp(gt,n)),('tail_cleanup',exact_n_distant_fp(gt,n))]:
            met=case_metrics(pred.astype(np.float32),gt.astype(np.float32),0.5,bf1_tolerance=2); f=metric_profile_features(met)
            rows.append({'case_id':str(meta['case_id']),'target_route':label,'matched_fp_budget':n,**met,**f,'scalar_quality':scalar_quality_from_features(f)})
    df=pd.DataFrame(rows); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    paired=df.pivot(index='case_id',columns='target_route',values='fp'); mismatch=(paired.max(axis=1)-paired.min(axis=1)).abs() if len(paired) else pd.Series(dtype=float)
    summary={'split':args.split,'n_rows':len(df),'n_cases':df.case_id.nunique(),'max_fp_count_mismatch':float(mismatch.max()) if len(mismatch) else None,
             'purpose':'Hold FP/FN volume constant while changing spatial geometry, directly testing whether FP/FN amount alone is insufficient.'}
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(df.groupby('target_route')[['fp','fn','dice','bf1','hd95','assd']].mean().to_string()); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
