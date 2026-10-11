from __future__ import annotations
import argparse,copy,datetime as dt,json,os,subprocess,sys,time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent

def stamp(): return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def load_yaml(p): return yaml.safe_load(Path(p).read_text())
def write_json(p,o): p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2))
def run(cmd,stage,expected=None,resume=True):
    expected=expected or []
    if resume and expected and all(Path(x).exists() for x in expected): print(f'[{stamp()}] SKIP {stage}',flush=True); return
    print(f'[{stamp()}] ==== {stage} ====',flush=True); print('$ '+' '.join(map(str,cmd)),flush=True); env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''); t=time.time(); subprocess.run(cmd,cwd=ROOT,env=env,check=True); print(f'[{stamp()}] seconds={time.time()-t:.1f}',flush=True)
def deep_update(a,b):
    o=copy.deepcopy(a)
    for k,v in (b or {}).items(): o[k]=deep_update(o[k],v) if isinstance(v,dict) and isinstance(o.get(k),dict) else copy.deepcopy(v)
    return o
def make_cfg(base,updates,path): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(deep_update(base,updates),sort_keys=False)); return path

def eval_std(ck,out,split,device,manifest,data_root,resume):
    out=Path(out); csv=out/'per_case_metrics.csv'; run([sys.executable,'-u','evaluate.py','--checkpoint',str(ck),'--manifest',manifest,'--data-root',data_root,'--split',split,'--device',device,'--output-dir',str(out)],f'EVAL {split} standard',[csv],resume); return csv
def eval_v3(ck,out,split,device,manifest,data_root,resume,policy=None,col=None):
    out=Path(out); csv=out/'per_case_metrics.csv'; cmd=[sys.executable,'-u','evaluate_mdcr_v3.py','--checkpoint',str(ck),'--manifest',manifest,'--data-root',data_root,'--split',split,'--device',device,'--output-dir',str(out)]
    if policy is not None: cmd += ['--force-policy-csv',str(policy),'--force-route-column',str(col)]
    run(cmd,f'EVAL {split} MDCR-v3'+(f' forced={col}' if col else ''),[csv],resume); return csv
def compare(a,b,an,bn,out,seed,resume): run([sys.executable,'-u','scripts/compare_two.py','--a',str(a),'--b',str(b),'--a-name',an,'--b-name',bn,'--output',str(out),'--seed',str(seed)],f'PAIR {bn} vs {an}',[Path(out)],resume)

def train_one(arch,seed,suite,cfg_path,cfg,root,device,resume,ablations=False):
    manifest=str(suite['manifest']); data_root=str(suite['data_root']); ad=root/'architectures'/arch/f'seed{seed}'; ad.mkdir(parents=True,exist_ok=True); bcfg=ROOT/str(suite['baseline_configs'][arch])
    bd=ad/'baseline'; bck=bd/'best.pt'; bm=bd/'_TRAIN_COMPLETE.json'
    if not(resume and bck.exists() and bm.exists()): run([sys.executable,'-u','train.py','--config',str(bcfg),'--manifest',manifest,'--data-root',data_root,'--device',device,'--seed',str(seed),'--output-dir',str(bd)],f'BASELINE {arch} {seed}',resume=False); write_json(bm,{'complete':True})
    cd=ad/'control'; cck=cd/'best_dice.pt'; cm=cd/'_TRAIN_COMPLETE.json'
    if not(resume and cck.exists() and cm.exists()): run([sys.executable,'-u','train_control.py','--config',str(cfg_path),'--manifest',manifest,'--data-root',data_root,'--device',device,'--seed',str(seed),'--checkpoint',str(bck),'--output-dir',str(cd)],f'CONTROL {arch} {seed}',resume=False); write_json(cm,{'complete':True})
    er=ad/'eval_val'; bval=eval_std(bck,er/'baseline','val',device,manifest,data_root,resume); cval=eval_std(cck,er/'control','val',device,manifest,data_root,resume)
    nat=suite.get('natural',{}) or {}; main=int(suite.get('primary_seeds',[42])[0]); mc=int(nat.get('mc_samples_main_seed',8) if (arch==suite['primary_architecture'] and seed==main) else nat.get('mc_samples_other_seeds',1)); diag=ad/'diagnostics'; trraw=diag/'natural_train_raw.csv'; varaw=diag/'natural_val_raw.csv'; common=['--checkpoint',str(bck),'--manifest',manifest,'--data-root',data_root,'--device',device,'--min-oracle-gain',str(nat.get('min_oracle_gain',0.002)),'--hd95-scale',str(nat.get('hd95_scale',16.0)),'--assd-scale',str(nat.get('assd_scale',4.0))]
    run([sys.executable,'-u','scripts/build_natural_correction_benchmark.py',*common,'--split','train','--mc-samples',str(mc),'--output',str(trraw)],f'DIAGNOSTIC TRAIN {arch} {seed}',[trraw],resume); run([sys.executable,'-u','scripts/build_natural_correction_benchmark.py',*common,'--split','val','--mc-samples',str(mc),'--output',str(varaw)],f'DIAGNOSTIC VAL {arch} {seed}',[varaw],resume)
    traug=ad/'route_policy_train_aug.csv'; vaaug=ad/'route_policy_val_aug.csv'; dr=diag/'route_prediction'; fs=list(suite.get('route_feature_sets',[]));
    if mc>1 and 'uncertainty_mc' not in fs: fs.append('uncertainty_mc')
    run([sys.executable,'-u','scripts/evaluate_route_diagnostics.py','--train-csv',str(trraw),'--test-csv',str(varaw),'--label-column','oracle_route','--output-dir',str(dr),'--feature-sets',*fs,'--augment-train-output',str(traug),'--augment-test-output',str(vaaug)],f'ROUTE DIAGNOSTICS {arch} {seed}',[dr/'route_diagnostic_summary.csv',traug,vaaug],resume)
    rd=ad/'router_v3'; rck=rd/'best.pt'; rm=rd/'_TRAIN_COMPLETE.json'
    if not(resume and rck.exists() and rm.exists()): run([sys.executable,'-u','train_router_v3.py','--config',str(cfg_path),'--source-checkpoint',str(bck),'--train-policy-csv',str(traug),'--val-policy-csv',str(vaaug),'--route-column',str(suite.get('router_train_route_column','oracle_route')),'--val-route-column',str(suite.get('router_val_route_column','oracle_route')),'--manifest',manifest,'--data-root',data_root,'--device',device,'--seed',str(seed),'--output-dir',str(rd)],f'HIERARCHICAL ROUTER {arch} {seed}',resume=False); write_json(rm,{'complete':True})
    re=ad/'router_v3_eval'; run([sys.executable,'-u','scripts/evaluate_router_v3.py','--router-checkpoint',str(rck),'--source-checkpoint',str(bck),'--policy-csv',str(vaaug),'--label-columns','oracle_route','route_metric_discordance_plus_fpfn','--manifest',manifest,'--data-root',data_root,'--split','val','--device',device,'--output-dir',str(re)],f'ROUTER EVAL {arch} {seed}',[re/'summary.csv'],resume)
    md=ad/'mdcr_v3'; dep=md/'deployment.pt'; om=md/'best_oracle_utility.pt'; mm=md/'_TRAIN_COMPLETE.json'
    if not(resume and dep.exists() and om.exists() and mm.exists()): run([sys.executable,'-u','train_mdcr_v3.py','--config',str(cfg_path),'--source-checkpoint',str(bck),'--router-checkpoint',str(rck),'--policy-csv',str(traug),'--val-policy-csv',str(vaaug),'--expert-route-column',str(suite.get('expert_train_route_column','oracle_route')),'--oracle-route-column',str(suite.get('oracle_route_column','oracle_route')),'--manifest',manifest,'--data-root',data_root,'--device',device,'--seed',str(seed),'--output-dir',str(md)],f'EXPLICIT EXPERTS {arch} {seed}',resume=False); write_json(mm,{'complete':True})
    learned=eval_v3(dep,er/'mdcr_v3_learned','val',device,manifest,data_root,resume); oracle=eval_v3(om,ad/'upper_bounds'/'true_oracle_raw','val',device,manifest,data_root,resume,vaaug,suite.get('oracle_route_column','oracle_route'))
    compare(bval,learned,'baseline','mdcr_v3',ad/'comparisons'/'mdcr_v3_vs_baseline.csv',seed,resume); compare(cval,learned,'control','mdcr_v3',ad/'comparisons'/'mdcr_v3_vs_control.csv',seed,resume); compare(cval,oracle,'control','true_oracle_raw',ad/'comparisons'/'oracle_raw_vs_control.csv',seed,resume)
    # Metric-teacher routing uses the same raw expert checkpoint, so it measures route quality without fallback contamination.
    mt=eval_v3(om,ad/'upper_bounds'/'metric_teacher_raw','val',device,manifest,data_root,resume,vaaug,'route_metric_discordance_plus_fpfn')
    run([sys.executable,'-u','scripts/final_round_decision.py','--baseline',str(er/'baseline'/'summary.json'),'--control',str(er/'control'/'summary.json'),'--learned',str(er/'mdcr_v3_learned'/'summary.json'),'--oracle-raw',str(ad/'upper_bounds'/'true_oracle_raw'/'summary.json'),'--output',str(ad/'FINAL_ROUND_DECISION.json')],f'FINAL DECISION {arch} {seed}',[ad/'FINAL_ROUND_DECISION.json'],resume=False)
    harm=ad/'harm_val_v3'; run([sys.executable,'-u','scripts/analyze_correction_harm_v3.py','--source-checkpoint',str(bck),'--refined-checkpoint',str(dep),'--manifest',manifest,'--data-root',data_root,'--split','val','--device',device,'--output-dir',str(harm)],f'HARM {arch} {seed}',[harm/'summary.json'],resume)
    if ablations and suite.get('run_seed42_ablations',True):
        gen=root/'generated_configs'
        for name,upd in (suite.get('expert_ablations',{}) or {}).items():
            acfg=make_cfg(cfg,upd,gen/f'mdcr_v3_{name}.yaml'); xd=root/'ablations'/'experts'/name; xdep=xd/'deployment.pt'; xm=xd/'_TRAIN_COMPLETE.json'
            if not(resume and xdep.exists() and xm.exists()): run([sys.executable,'-u','train_mdcr_v3.py','--config',str(acfg),'--source-checkpoint',str(bck),'--router-checkpoint',str(rck),'--policy-csv',str(traug),'--val-policy-csv',str(vaaug),'--expert-route-column','oracle_route','--oracle-route-column','oracle_route','--manifest',manifest,'--data-root',data_root,'--device',device,'--seed',str(seed),'--output-dir',str(xd)],f'ABLATION {name}',resume=False); write_json(xm,{'complete':True})
            xv=eval_v3(xdep,xd/'eval_val','val',device,manifest,data_root,resume); compare(bval,xv,'baseline',name,xd/'vs_baseline.csv',seed,resume)
    return {'arch':arch,'seed':seed,'baseline':bck,'control':cck,'mdcr':dep}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--suite-config',default='configs/mdcr_v3_suite.yaml'); ap.add_argument('--device',default='auto'); ap.add_argument('--data-root',default=None); ap.add_argument('--manifest',default=None); ap.add_argument('--no-resume',action='store_true'); ap.add_argument('--skip-test',action='store_true'); a=ap.parse_args(); resume=not a.no_resume; suite=load_yaml(ROOT/a.suite_config)
    if a.data_root: suite['data_root']=a.data_root
    if a.manifest: suite['manifest']=a.manifest
    out=ROOT/str(suite['output_root']); out.mkdir(parents=True,exist_ok=True); manifest=str(suite['manifest']); data_root=str(suite['data_root']); cfgp=ROOT/str(suite['mdcr_v3_config']); cfg=load_yaml(cfgp); mp=Path(manifest); mp=mp if mp.is_absolute() else ROOT/mp
    if not mp.exists(): run([sys.executable,'-u','scripts/prepare_brisc2025.py','--data-root',data_root,'--manifest',manifest,'--prefer-backend','raw'],'PREPARE',[mp],resume=False)
    syn=suite.get('synthetic',{}) or {}; sd=out/'synthetic'; tr=sd/'train.csv'; va=sd/'val.csv'; run([sys.executable,'-u','scripts/build_synthetic_discordance_benchmark.py','--manifest',manifest,'--data-root',data_root,'--split','train','--output',str(tr),'--max-cases',str(syn.get('train_max_cases',600)),'--seed',str(syn.get('seed',4201))],'SYNTH TRAIN',[tr],resume); run([sys.executable,'-u','scripts/build_synthetic_discordance_benchmark.py','--manifest',manifest,'--data-root',data_root,'--split','val','--output',str(va),'--max-cases',str(syn.get('val_max_cases',300)),'--seed',str(int(syn.get('seed',4201))+1)],'SYNTH VAL',[va],resume); run([sys.executable,'-u','scripts/evaluate_route_diagnostics.py','--train-csv',str(tr),'--test-csv',str(va),'--label-column','target_route','--output-dir',str(sd/'diagnostics'),'--feature-sets','dice_only','fpfn','scalar_quality','metric_values','metric_discordance','metric_discordance_plus_fpfn'],'SYNTH DIAG',[sd/'diagnostics'/'route_diagnostic_summary.csv'],resume)
    mt=sd/'matched_train.csv'; mv=sd/'matched_val.csv'; run([sys.executable,'-u','scripts/build_matched_geometry_benchmark.py','--manifest',manifest,'--data-root',data_root,'--split','train','--output',str(mt),'--max-cases',str(syn.get('train_max_cases',600))],'MATCH TRAIN',[mt],resume); run([sys.executable,'-u','scripts/build_matched_geometry_benchmark.py','--manifest',manifest,'--data-root',data_root,'--split','val','--output',str(mv),'--max-cases',str(syn.get('val_max_cases',300))],'MATCH VAL',[mv],resume); run([sys.executable,'-u','scripts/evaluate_route_diagnostics.py','--train-csv',str(mt),'--test-csv',str(mv),'--label-column','target_route','--output-dir',str(sd/'matched_diagnostics'),'--feature-sets','fpfn','scalar_quality','metric_values','metric_discordance','metric_discordance_plus_fpfn'],'MATCH DIAG',[sd/'matched_diagnostics'/'route_diagnostic_summary.csv'],resume)
    rec=[]; pa=suite['primary_architecture']; ps=[int(x) for x in suite.get('primary_seeds',[])];
    for i,s in enumerate(ps): rec.append(train_one(pa,s,suite,cfgp,cfg,out,a.device,resume,ablations=(i==0)))
    ra=suite.get('replication_architecture');
    if ra:
        for s in [int(x) for x in suite.get('replication_seeds',[])]: rec.append(train_one(ra,s,suite,cfgp,cfg,out,a.device,resume,False))
    if suite.get('evaluate_official_test',True) and not a.skip_test:
        lt=out/'locked_test'
        for r in rec:
            td=lt/r['arch']/f"seed{r['seed']}"; bt=eval_std(r['baseline'],td/'baseline','test',a.device,manifest,data_root,resume); ct=eval_std(r['control'],td/'control','test',a.device,manifest,data_root,resume); md=eval_v3(r['mdcr'],td/'mdcr_v3_learned','test',a.device,manifest,data_root,resume); compare(bt,md,'baseline','mdcr_v3',td/'mdcr_v3_vs_baseline.csv',r['seed'],resume); compare(ct,md,'control','mdcr_v3',td/'mdcr_v3_vs_control.csv',r['seed'],resume)
    run([sys.executable,'-u','scripts/summarize_mdcr_v3_suite.py','--output-root',str(out)],'SUMMARIZE',[out/'analysis'/'performance_by_seed.csv'],resume=False); write_json(out/'SUITE_COMPLETE.json',{'complete':True,'finished_at':stamp(),'records':[{'arch':r['arch'],'seed':r['seed']} for r in rec]}); print(f'[{stamp()}] MDCR-v3 FINAL ROUND COMPLETE -> {out}',flush=True)
if __name__=='__main__': main()
