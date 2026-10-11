from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def rj(p): return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',required=True); a=ap.parse_args(); root=Path(a.output_root); out=root/'analysis'; out.mkdir(parents=True,exist_ok=True); perf=[]; routers=[]; upper=[]; decisions=[]
    for ad in (root/'architectures').glob('*') if (root/'architectures').exists() else []:
        for sd in ad.glob('seed*'):
            try: seed=int(sd.name.replace('seed',''))
            except: continue
            for method,rel in [('baseline','eval_val/baseline/summary.json'),('control','eval_val/control/summary.json'),('mdcr_v3','eval_val/mdcr_v3_learned/summary.json')]:
                p=sd/rel
                if p.exists():
                    j=rj(p); perf.append({'architecture':ad.name,'seed':seed,'method':method,'dice':j.get('dice_mean'),'bf1':j.get('bf1_mean'),'hd95':j.get('hd95_mean_finite'),'assd':j.get('assd_mean_finite'),'misses':j.get('missed_lesion_cases')})
            rp=sd/'router_v3/router_summary.json'
            if rp.exists():
                j=rj(rp); v=j.get('best_validation',{}); routers.append({'architecture':ad.name,'seed':seed,'accuracy':v.get('accuracy'),'balanced_accuracy':v.get('balanced_accuracy'),'macro_f1':v.get('macro_f1'),**{f'recall_{k}':x for k,x in (v.get('per_route_recall',{}) or {}).items()}})
            for name in ['true_oracle_raw','metric_teacher_raw']:
                p=sd/'upper_bounds'/name/'summary.json'
                if p.exists():
                    j=rj(p); upper.append({'architecture':ad.name,'seed':seed,'routing':name,'dice':j.get('dice_mean'),'bf1':j.get('bf1_mean'),'hd95':j.get('hd95_mean_finite'),'assd':j.get('assd_mean_finite')})
            p=sd/'FINAL_ROUND_DECISION.json'
            if p.exists(): decisions.append({'architecture':ad.name,'seed':seed,**rj(p)})
    pf=pd.DataFrame(perf); pf.to_csv(out/'performance_by_seed.csv',index=False)
    if len(pf): pf.groupby(['architecture','method'])[['dice','bf1','hd95','assd','misses']].agg(['mean','std']).reset_index().to_csv(out/'performance_multiseed_summary.csv',index=False)
    pd.DataFrame(routers).to_csv(out/'router_by_seed.csv',index=False); pd.DataFrame(upper).to_csv(out/'raw_oracle_ceiling_by_seed.csv',index=False)
    # Flatten final decision essentials.
    dr=[]
    for x in decisions:
        dr.append({'architecture':x['architecture'],'seed':x['seed'],'decision':x.get('decision'),'oracle_utility_gain_vs_control':x.get('oracle_raw_vs_control',{}).get('utility_gain'),'oracle_dice_gain_vs_control':x.get('oracle_raw_vs_control',{}).get('dice_gain'),'oracle_hd95_reduction_vs_control':x.get('oracle_raw_vs_control',{}).get('hd95_reduction_px'),'oracle_dice_gap_vs_learned':x.get('oracle_raw_vs_learned',{}).get('dice_gap')})
    pd.DataFrame(dr).to_csv(out/'final_round_decisions.csv',index=False)
    tr=[]; lt=root/'locked_test'
    if lt.exists():
        for ad in lt.glob('*'):
            for sd in ad.glob('seed*'):
                try: seed=int(sd.name.replace('seed',''))
                except: continue
                for method,rel in [('baseline','baseline/summary.json'),('control','control/summary.json'),('mdcr_v3','mdcr_v3_learned/summary.json')]:
                    p=sd/rel
                    if p.exists():
                        j=rj(p); tr.append({'architecture':ad.name,'seed':seed,'method':method,'dice':j.get('dice_mean'),'bf1':j.get('bf1_mean'),'hd95':j.get('hd95_mean_finite'),'assd':j.get('assd_mean_finite'),'misses':j.get('missed_lesion_cases')})
    t=pd.DataFrame(tr); t.to_csv(out/'locked_test_by_seed.csv',index=False)
    if len(t): t.groupby(['architecture','method'])[['dice','bf1','hd95','assd','misses']].agg(['mean','std']).reset_index().to_csv(out/'locked_test_multiseed_summary.csv',index=False)
    print(f'Wrote MDCR-v3 final-round tables to {out}')
if __name__=='__main__': main()
