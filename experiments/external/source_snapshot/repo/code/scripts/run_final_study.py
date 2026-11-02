"""Resumable posture study; the historical corrected benchmark stays frozen."""
from __future__ import annotations
import argparse,json,hashlib,sys,time
from pathlib import Path
import portable_lock as fcntl
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
from spine_registration.benchmark import ImageProblem,atomic_json,evaluate,fingerprint,global_to_local,load_case
from spine_registration.benchmark_optimizers import optimize_powell,optimize_torch
from spine_registration.torch_registration import motion_penalty_torch,ncc_loss_torch
from posture_geometry import prepare_geometry
from generate_posture_dataset import field_of_view,plain
from final_study_support import articulated,sha256,extra_metrics,degrade,matrices
from batched_global_search import coarse_cma
from scipy.optimize import minimize
OUTPUT=ROOT/'outputs/final_study'

def configuration():
    manifest=json.loads((ROOT/'inputs/ppcnet75/manifest.json').read_text());cases=[r['case'] for r in manifest['cases']]
    rng=np.random.default_rng(20261024)
    auxiliary=sorted(rng.choice(cases,20,replace=False).tolist());sensitivity=sorted(rng.choice(cases,15,replace=False).tolist())
    files=sorted((ROOT/'code/src/spine_registration').glob('*.py'))
    files += [Path(__file__),Path(__file__).with_name('final_study_support.py'),Path(__file__).with_name('active_ray_renderer.py'),Path(__file__).with_name('batched_global_search.py'),Path(__file__).with_name('posture_geometry.py'),Path(__file__).with_name('generate_posture_dataset.py')]
    return dict(version='final_study_v2.2',seed=20261024,cases=cases,engineering_cases=cases[:12],auxiliary_cases=auxiliary,sensitivity_cases=sensitivity,
        input_manifest_hash=fingerprint(manifest),source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
        levels=[64,128,256],pixel_spacing_mm=2.,sample_step_mm=1.,max_samples=192,chunk_rows=64,
        target_resolution=512,target_sample_step_mm=.5,target_max_samples=384,lambda_motion=.1,
        endpoint_angles_deg={'rigid':0.,'small':2.,'moderate':8.,'large':16.},
        motion_sign='case index balanced, shared across amplitudes',occluded_endpoint='L1/L5 by floor(case index/2) parity, crossed with motion sign',segment_weights='U(.8,1.2), normalized over four joints',
        global_motion='rotvec components U(-3,3) deg; translation components U(-5,5) mm; L5 pivot',
        joint_separation='smallest .1 mm step in [0,2] passing geometry',
        geometry_rule='penetration beyond largest voxel=0; overlap<=1%; centroid ratio .9-1.1; detector margin6px',
        interpretation='synthetic geometric scenarios, not clinical six-DoF distributions or measured Cobb angles',
        A=dict(population=32,cma_sigma=5.,cma_tolx=1e-4,cma_tolfun=1e-5,partition_starts=8,rotation_bound_deg=20.,translation_bound_mm=40.,max_iterations=500,xtol=1e-4,ftol=1e-5),
        BC=dict(max_iterations=500,adam_warmup=30,adam_lr=.2,relative_tolerance=1e-4,relative_patience=5,gradient_tolerance=1e-5),
        A_polish='image-only LBFGS, same levels, no Adam, choose by image objective',
        primary_scenario='combined',primary_metric='surface_rms_tre_mm',bootstrap_replicates=10000,
        exposed_data='All 75 cases previously inspected; no untouched confirmatory test set claimed')

def scenario_selected(scenario,name,config):
    if scenario=='rigid':return name in config['engineering_cases']
    if scenario in ['combined_r1','combined_r2']:return name in config['auxiliary_cases']
    return True

def frozen_config():
    config=configuration();path=OUTPUT/'config.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise ValueError('Source/config changed: use a new output version')
    else:atomic_json(path,config)
    return config,fingerprint(config)

def motion_for(case,prepared,index,config,kind):
    rng=np.random.default_rng(config['seed']+index*1009)
    rotation=np.deg2rad(rng.uniform(-3,3,3));translation=rng.uniform(-5,5,3)
    weights=rng.uniform(.8,1.2,4);weights/=weights.sum();sign=1 if index%2==0 else -1
    angles=weights*config['endpoint_angles_deg'].get(kind,0.)*sign
    if kind=='focal':
        k=config['auxiliary_cases'].index(case['case']);angles=np.eye(4)[k%4]*6.*(1 if (k//4)%2==0 else -1)
    motion=articulated(prepared,angles,rotation,translation)
    motion['field_of_view']=field_of_view(case,motion,ImageProblem(case,config).projections)
    motion['accepted'] &= bool(motion['field_of_view']['accepted'])
    return motion

def save_targets(directory,targets,metadata):
    directory.mkdir(parents=True,exist_ok=True);arrays={v:t.detach().cpu().numpy() for v,t in targets.items()}
    metadata['image_hashes']={v:hashlib.sha256(a.tobytes()).hexdigest() for v,a in arrays.items()}
    if (directory/'metadata.json').exists() and json.loads((directory/'metadata.json').read_text())!=plain(metadata):
        raise ValueError(f'Incompatible target {directory}')
    for view,array in arrays.items():
        path=directory/(view+'.npy')
        if path.exists():
            if not np.array_equal(np.load(path,allow_pickle=False),array):raise ValueError('Target changed')
        else:
            with path.with_suffix('.tmp').open('wb') as handle:np.save(handle,array)
            path.with_suffix('.tmp').replace(path)
    atomic_json(directory/'metadata.json',plain(metadata))

def verified_case(name):
    directory=ROOT/'inputs/ppcnet75'/name
    manifest=json.loads((ROOT/'inputs/ppcnet75/manifest.json').read_text())
    row=next(r for r in manifest['cases'] if r['case']==name)
    for filename,key in [('centred_case.npz','volume_sha256'),('motion_ground_truth.json','gt_sha256')]:
        if sha256(directory/filename)!=row[key]:raise ValueError('Original input checksum changed')
    return load_case(directory)

def prepare_case(name,index,config,config_hash):
    base=ROOT/'inputs/ppcnet75'/name;case=verified_case(name)
    with np.load(base/'centred_case.npz') as archive:prepared=prepare_geometry(archive['segmentation'],archive['affine_mm'])
    generator=ImageProblem(case,{**config,'sample_step_mm':config['target_sample_step_mm'],'max_samples':config['target_max_samples']})
    kinds=list(config['endpoint_angles_deg'])+(['focal'] if name in config['auxiliary_cases'] else [])
    for kind in kinds:
        directory=OUTPUT/'targets'/kind/name
        if (directory/'metadata.json').exists():continue
        motion=motion_for(case,prepared,index,config,kind)
        metadata=dict(case=name,scenario=kind,config_hash=config_hash,motion=plain(motion),centres=case['centres'].tolist(),accepted=motion['accepted'])
        if not motion['accepted']:atomic_json(directory/'metadata.json',metadata);continue
        with torch.no_grad():
            images=generator.parts(torch.tensor(motion['params'],dtype=torch.float32,device='cuda'),512)
            target={v:F.interpolate(torch.stack(parts).sum(0)[None,None],size=(256,256),mode='area')[0,0] for v,parts in images.items()}
        save_targets(directory,target,metadata)
    directory=OUTPUT/'targets/moderate'/name;meta=json.loads((directory/'metadata.json').read_text())
    if meta['accepted']:
        target={v:torch.tensor(np.load(directory/(v+'.npy')),device='cuda') for v in ['AP','LAT']}
        with torch.no_grad():parts=generator.parts(torch.tensor(meta['motion']['params'],dtype=torch.float32,device='cuda'),256)
        scales={v:float(np.log(20.)/np.quantile(x.cpu().numpy()[x.cpu().numpy()>0],.995)) for v,x in target.items()}
        definitions=[('combined',20000,.003,.15,0),('combined_r1',20000,.003,.15,1),('combined_r2',20000,.003,.15,2)]
        if name in config['auxiliary_cases']:
            definitions += [('gaussian',None,.003,0.,0),('poisson',20000,0.,0.,0),('occlusion15',None,0.,.15,0),('occlusion30',None,0.,.30,0)]
        for scenario,photons,gaussian,occlusion,repeat in definitions:
            out=OUTPUT/'targets'/scenario/name
            if (out/'metadata.json').exists():continue
            observations,corruption=degrade(target,parts,seed=config['seed']+index*1009+repeat*12347,
                photons=photons,gaussian=gaussian,occlusion=occlusion,endpoint=0 if (index//2)%2==0 else 4,scales=scales)
            save_targets(out,observations,{**meta,'scenario':scenario,'base_scenario':'moderate','corruption':corruption,'repeat':repeat})
    return {k:json.loads((OUTPUT/'targets'/k/name/'metadata.json').read_text())['accepted'] for k in kinds}

def read_target(name,scenario,config_hash):
    directory=OUTPUT/'targets'/scenario/name
    if not (directory/'metadata.json').exists():return None,None
    meta=json.loads((directory/'metadata.json').read_text())
    if meta['config_hash']!=config_hash:raise ValueError('Target config mismatch')
    if not meta['accepted']:return None,meta
    targets={}
    for view in ['AP','LAT']:
        array=np.load(directory/(view+'.npy'),allow_pickle=False)
        if hashlib.sha256(array.tobytes()).hexdigest()!=meta['image_hashes'][view]:raise ValueError('Target hash mismatch')
        targets[view]=torch.as_tensor(array,device='cuda')
    return targets,meta

def local_optimize(problem,initial,weight,config,views=None,hard=False):
    pose=torch.as_tensor(initial,dtype=torch.float32,device='cuda');levels=[]
    original_dist=torch.linalg.vector_norm(problem.centres[:-1]-problem.centres[1:],dim=-1)
    for level in config['levels']:
        def objective(p):
            if views is None:image=problem.image_loss(p,level)
            else:
                parts=problem.parts(p,level)
                image=torch.stack([ncc_loss_torch(problem.targets[level][v],torch.stack(parts[v]).sum(0)) for v in views]).mean()
            prior=motion_penalty_torch(p,centres=problem.centres,sigma_rotation_deg=3.,sigma_translation_mm=3.)
            value=image+weight*prior
            if hard:
                moved=problem.centres+p[:,3:]
                distances=torch.linalg.vector_norm(moved[:-1]-moved[1:],dim=-1)
                value+=1000*torch.relu(.8*original_dist-distances).square().mean()
            return value
        result=optimize_torch(objective,pose,mode='joint',**config['BC'])
        pose=result.pop('pose');levels.append(dict(level=level,**plain(result)))
    return pose.detach().cpu().numpy(),dict(levels=levels,iterations=sum(x['iterations'] for x in levels),
        evaluations=sum(x['evaluations'] for x in levels),stop_reason=levels[-1]['stop_reason'],
        converged=levels[-1]['converged'],hit_max_iterations=any(x['hit_max_iterations'] for x in levels))

def optimize_A(problem,config):
    cfg=config['A'];seed=config['seed']+config['cases'].index(problem.case['case'])*1009
    coarse=coarse_cma(problem,seed=seed,starts=cfg['partition_starts'],population=cfg['population'],max_iterations=cfg['max_iterations'])
    pose=np.asarray(coarse['pose']);refinements=[];polish=[]
    scale=np.array([np.deg2rad(.01)]*3+[1.]*3)
    bounds=np.array([np.deg2rad(cfg['rotation_bound_deg'])]*3+[cfg['translation_bound_mm']]*3)
    for level in config['levels']:
        objective=problem.powell_objective(level);history=[];initial=pose.copy()
        previous=dict(q=pose/scale,loss=objective(pose))
        def function(q):return objective(q*scale)
        def callback(q):
            value=function(q);step=float(np.max(np.abs(q-previous['q'])))
            history.append(dict(loss=value,normalized_step_inf=step,objective_decrease=previous['loss']-value))
            previous.update(q=q.copy(),loss=value)
            if step<cfg['xtol']:raise StopIteration
        result=minimize(function,pose/scale,method='Powell',bounds=list(zip(-bounds/scale,bounds/scale)),callback=callback,
            options=dict(xtol=cfg['xtol'],ftol=cfg['ftol'],maxiter=cfg['max_iterations']))
        if float(result.fun)<objective(initial):pose=result.x*scale
        reason={0:'metric_tolerance',2:'max_iterations',99:'step_tolerance'}.get(int(result.status),'optimizer_failure')
        refinements.append(dict(level=level,iterations=int(result.nit),evaluations=int(result.nfev),stop_reason=reason,
            hit_max_iterations=int(result.nit)>=cfg['max_iterations'],history=history,message=str(result.message)))
        trial=optimize_torch(lambda p:problem.image_loss(p,level,True),
            torch.tensor(pose,dtype=torch.float32,device='cuda'),mode='joint',**{**config['BC'],'adam_warmup':0})
        candidate=trial.pop('pose').detach().cpu().numpy()
        if objective(candidate)<=objective(pose):pose=candidate
        polish.append(dict(level=level,**plain(trial)))
    return pose,dict(cma=coarse,powell=refinements,polish=polish,
        iterations=coarse['iterations']+sum(x['iterations'] for x in refinements+polish),
        evaluations=coarse['evaluations']+sum(x['evaluations'] for x in refinements+polish),
        stop_reason=polish[-1]['stop_reason'],converged=polish[-1]['converged'],
        hit_max_iterations=any(x['hit_max_iterations'] for x in coarse['diagnostics']+refinements+polish))

def run_case(name,config,config_hash,scenarios,methods):
    case=verified_case(name);problem=ImageProblem(case,config);index=config['cases'].index(name)
    for scenario in scenarios:
        if not scenario_selected(scenario,name,config):continue
        base='moderate' if scenario in ['AP_only','LAT_only','initialization'] else scenario
        targets,meta=read_target(name,base,config_hash)
        if targets is None:continue
        case['gt_params']=np.asarray(meta['motion']['params']);case['gt_matrices']=np.asarray(meta['motion']['transforms'])
        problem.set_targets(targets)
        for method in methods:
            if scenario in ['AP_only','LAT_only','initialization'] and method not in ['B','C']:continue
            directory=OUTPUT/'results'/scenario/name;directory.mkdir(parents=True,exist_ok=True)
            path=directory/(method+'.json');target_hash=fingerprint(meta)
            if path.exists():
                existing=json.loads(path.read_text())
                if existing['config_hash']!=config_hash or existing['target_hash']!=target_hash:raise ValueError('Incompatible result checkpoint')
                continue
            torch.cuda.synchronize();started=time.monotonic()
            if method=='Zero':
                pose=np.zeros((5,6));diagnostics=dict(iterations=0,evaluations=0,stop_reason='not_optimized',converged=None,hit_max_iterations=False)
            elif method=='A':pose,diagnostics=optimize_A(problem,config)
            else:
                apath=OUTPUT/'results'/base/name/'A.json'
                if not apath.exists():raise ValueError(f'A initialization missing: {apath}')
                initial=global_to_local(np.asarray(json.loads(apath.read_text())['pose']),case['centres'])
                if scenario=='initialization':
                    rng=np.random.default_rng(config['seed']+index*319)
                    axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
                    shift=rng.normal(size=3);shift*=10/np.linalg.norm(shift)
                    perturb=Rotation.from_rotvec(np.deg2rad(10)*axis).as_matrix()
                    old=matrices(initial,case['centres']);anchor=case['centres'].mean(0)
                    g=np.eye(4);g[:3,:3]=perturb;g[:3,3]=anchor+shift-perturb@anchor
                    moved=g[None]@old
                    initial=np.column_stack((Rotation.from_matrix(moved[:,:3,:3]).as_rotvec(),
                        np.einsum('nij,nj->ni',moved[:,:3,:3],case['centres'])+moved[:,:3,3]-case['centres']))
                weight=.1 if method=='C' else float(method.split('_')[1]) if method.startswith('lambda_') else 0.
                views=[scenario.split('_')[0]] if scenario in ['AP_only','LAT_only'] else None
                pose,diagnostics=local_optimize(problem,initial,weight,config,views,hard=method=='E')
                diagnostics['initial_params']=initial.tolist();diagnostics['lambda_motion']=weight
            torch.cuda.synchronize();seconds=time.monotonic()-started
            report=evaluate(pose,case,'A' if method=='A' else 'Zero' if method=='Zero' else 'B')
            extra=extra_metrics(pose,case,method=='A')
            for row,additional in zip(report['vertebrae'],extra):row.update(additional)
            for key in extra[0]:report['metrics'][key]=float(np.mean([x[key] for x in extra]))
            with torch.no_grad():
                tensor=torch.as_tensor(pose,device='cuda',dtype=torch.float32)
                loss=float(problem.image_loss(tensor,256,method=='A'))
                local=global_to_local(pose,case['centres']) if method=='A' else pose
                penalty=float(motion_penalty_torch(torch.tensor(local,device='cuda',dtype=torch.float32),centres=problem.centres))
            result=dict(case=name,scenario=scenario,method=method,config_hash=config_hash,target_hash=target_hash,
                pose=pose.tolist(),runtime_seconds=seconds,ncc_loss=loss,motion_penalty=penalty,**report,optimization=diagnostics)
            atomic_json(path,plain(result))
            print(json.dumps(dict(case=name,scenario=scenario,method=method,seconds=round(seconds,2),
                translation=round(report['metrics']['translation_error_mm'],4),rotation=round(report['metrics']['rotation_error_deg'],4),
                stop=diagnostics['stop_reason'])),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--case');parser.add_argument('--limit',type=int);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--scenarios',nargs='+',default=['rigid','small','moderate','large','focal','gaussian','poisson','occlusion15','occlusion30','combined','combined_r1','combined_r2'])
    parser.add_argument('--methods',nargs='+',default=['Zero','A','B','C']);args=parser.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards:parser.error('Invalid shard')
    valid={'Zero','A','B','C','E','lambda_0.01','lambda_0.05','lambda_0.2','lambda_0.5','lambda_1.0'}
    if any(m not in valid for m in args.methods):parser.error('Unsupported method')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUTPUT.mkdir(parents=True,exist_ok=True);config,config_hash=frozen_config()
    lock=(OUTPUT/(args.action+f'_shard{args.shard}.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=[x for i,x in enumerate(config['cases']) if i%args.shards==args.shard]
    if args.case:names=[args.case]
    if args.limit:names=names[:args.limit]
    for name in names:
        if args.action=='prepare':
            status=prepare_case(name,config['cases'].index(name),config,config_hash);print(name,status,flush=True)
        else:run_case(name,config,config_hash,args.scenarios,args.methods)
    atomic_json(OUTPUT/(args.action+f'_shard{args.shard}_completion.json'),
                dict(cases=names,config_hash=config_hash,scenarios=args.scenarios,methods=args.methods,status='COMPLETE'))

if __name__=='__main__':main()
