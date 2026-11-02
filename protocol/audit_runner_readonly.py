"""Static scientific-method audit. Reads source only; imports no runner/GPU code."""
from pathlib import Path
import ast,hashlib,json,datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TASK=ROOT.parents[1]
OLD=TASK/'work/repro_pkg/retained_source_bundle/code'
NEW=ROOT/'repo/code'
OUT=HERE/'audits';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nodes(p):
    tree=ast.parse(p.read_text(encoding='utf8'))
    result={}
    for x in tree.body:
        if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            result[x.name]=ast.dump(x,include_attributes=False)
    return result
changed=[];same=0
for p in OLD.rglob('*.py'):
    q=NEW/p.relative_to(OLD)
    if not q.exists():continue
    if sha(p)==sha(q):same+=1
    else:
        a,b=nodes(p),nodes(q)
        changed.append({'file':p.relative_to(OLD).as_posix(),'old_sha256':sha(p),'new_sha256':sha(q),'changed_top_level_definitions':[k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]})
targets={
 'scripts/adaptive_study.py':['relative_frames','edge_penalties','dispersion','perturb','refine','record'],
 'scripts/run_final_study.py':['motion_for','optimize_A'],
 'scripts/batched_global_search.py':['coarse_cma','batch_objective'],
 'scripts/final_study_support.py':['degrade','extra_metrics'],
 'scripts/adaptive_siddon.py':['render'],
 'src/spine_registration/benchmark.py':['ImageProblem','evaluate','global_to_local','projections'],
 'src/spine_registration/benchmark_optimizers.py':['optimize_torch'],
 'src/spine_registration/torch_registration.py':['ncc_loss_torch','TorchConeBeamRenderer'],
}
checks=[]
for rel,names in targets.items():
    a,b=nodes(OLD/rel),nodes(NEW/rel)
    for name in names:
        if name not in a or name not in b:checks.append({'file':rel,'definition':name,'status':'MISSING'});continue
        checks.append({'file':rel,'definition':name,'status':'IDENTICAL_AST' if a[name]==b[name] else 'CHANGED'})
runner=ROOT/'runner/run_siddon_validation.py'
selection=json.loads((ROOT/'repo/outputs/adaptive_prior_study/calibration/selection.json').read_text())
report={'review_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_type':'read-only static source and configuration review; no import/execution of GPU or outcome files','status':'PASS_CORE_METHODS_WITH_PROVENANCE_FOLLOWUPS','runner_sha256':sha(runner),'unchanged_source_files':same,'changed_files':changed,'core_function_checks':checks,'adaptive_values':selection['adaptive'],'primary_sources':{'old_code':str(OLD),'new_code':str(NEW)},'scope':'Legacy-cohort pilot runner only; external cohort path not yet implemented/audited','followups':['Save masks as arrays or make reconstruction from bounds explicit, with plan hash and persisted per-view RNG-state hashes.','Legacy corruption-support cache needs a source/config/array checksum sidecar checked on reuse; current cache path existence alone does not validate its contents.','Full production invocation must freeze the complete cohort manifest, six scenarios and protocol hash; a pilot command defaults to only clean and combined_s0.','External implementation must use new base seeds 20361024/20361025 and final assigned case indices0..49, with no geometry-based renumbering.'],'interpretation':'Target uses new independent Siddon512-to256 images, and all final comparators use a new common A/candidate bank. Shared trilinear references define corruption masks/scales/RNG, by root decision, and must be disclosed.'}
assert all(x['status']=='IDENTICAL_AST' for x in checks),checks
assert selection['adaptive']=={'lambda_max':0.01,'tau':0.01389625625827594}
(OUT/'runner_static_audit_v1.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'status':report['status'],'core_functions':len(checks),'unchanged_files':same,'changed_files':changed,'runner_sha256':report['runner_sha256']},indent=2))
