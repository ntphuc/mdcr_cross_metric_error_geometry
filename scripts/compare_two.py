from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS={"dice":"higher","iou":"higher","bf1":"higher","hd95":"lower","assd":"lower"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--a-name',required=True); ap.add_argument('--b-name',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    a=pd.read_csv(args.a); b=pd.read_csv(args.b); m=a.merge(b,on='case_id',suffixes=('_a','_b'),validate='one_to_one')
    rows=[]; rng=np.random.default_rng(args.seed)
    for metric,direction in METRICS.items():
        x=m[f'{metric}_a'].to_numpy(float); y=m[f'{metric}_b'].to_numpy(float); mask=np.isfinite(x)&np.isfinite(y); x,y=x[mask],y[mask]
        d=(y-x) if direction=='higher' else (x-y)
        if len(d):
            idx=rng.integers(0,len(d),size=(4000,len(d))); boot=d[idx].mean(1); lo,hi=np.quantile(boot,[.025,.975]); imp=d.mean()
            try: p=1.0 if np.allclose(d,0) else float(wilcoxon(d,alternative='two-sided',zero_method='wilcox').pvalue)
            except ValueError: p=np.nan
            wins=int((d>0).sum()); losses=int((d<0).sum())
        else: imp=lo=hi=p=np.nan; wins=losses=0
        dz=float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1)>1e-12 else 0.0
        rows.append({'comparison':f'{args.b_name} vs {args.a_name}','metric':metric,'direction':direction,'n':len(d),'improvement':imp,'ci95_low':lo,'ci95_high':hi,'wilcoxon_p':p,'cohens_dz_favorable':dz,'wins':wins,'losses':losses})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
