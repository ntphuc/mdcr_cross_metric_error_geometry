from __future__ import annotations
import argparse,json
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def utility(j):
    import math
    d=float(j['dice_mean']); b=float(j['bf1_mean']); h=j.get('hd95_mean_finite'); a=j.get('assd_mean_finite'); hs=16.0; ass=4.0
    he=math.exp(-float(h)/hs) if h is not None else 0.0; ae=math.exp(-float(a)/ass) if a is not None else 0.0
    return d+0.15*b+0.05*he+0.03*ae

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',required=True); ap.add_argument('--control',required=True); ap.add_argument('--learned',required=True); ap.add_argument('--oracle-raw',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    b,c,l,o=map(read,[a.baseline,a.control,a.learned,a.oracle_raw]); ub,uc,ul,uo=map(utility,[b,c,l,o]); dice_gain=float(o['dice_mean'])-float(c['dice_mean']); hd_gain=float(c['hd95_mean_finite'])-float(o['hd95_mean_finite']); router_gap=float(o['dice_mean'])-float(l['dice_mean'])
    oracle_beats_control=(uo>uc+1e-5) and (dice_gain>=-0.001)
    if oracle_beats_control and router_gap>0.0005: decision='ROUTER_BOTTLENECK'
    elif oracle_beats_control: decision='EXPERTS_HAVE_CEILING_ROUTER_GAP_SMALL'
    else: decision='EXPERT_BOTTLENECK_OR_NO_METHOD_CEILING'
    obj={'decision':decision,'utilities':{'baseline':ub,'control':uc,'learned':ul,'oracle_raw':uo},'oracle_raw_vs_control':{'dice_gain':dice_gain,'hd95_reduction_px':hd_gain,'utility_gain':uo-uc},'oracle_raw_vs_learned':{'dice_gap':router_gap,'utility_gap':uo-ul},'interpretation':'Oracle raw checkpoint is selected only among trained expert epochs (epoch>=1), before safety fallback. It is a diagnostic upper bound, never an official-test method.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(obj,indent=2)); print(json.dumps(obj,indent=2))
if __name__=='__main__': main()
