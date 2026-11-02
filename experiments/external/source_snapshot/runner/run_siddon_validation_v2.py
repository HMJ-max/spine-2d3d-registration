"""Independent Siddon targets, frozen matched registration, local resumable run.

Only native input paths, OS locking and GPU chunk sizes differ from the retained
optimization implementation. Previously fitted poses are never optimization starts.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os, platform, sys, time, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT/'repo'
TASK = ROOT.parents[1]
sys.path[:0] = [str(REPO/'dependencies'), str(TASK/'work/render_dependencies'),
               str(REPO/'code/scripts'), str(REPO/'code/src')]
import numpy as np
import scipy
import torch
from torch.nn import functional as F
from spine_registration.benchmark import ImageProblem, atomic_json, fingerprint, global_to_local, load_case
from spine_registration.drr import Volume
from spine_registration.projection import project_points
from adaptive_siddon import render as siddon
from adaptive_study import record, perturb, dispersion
from final_study_support import sha256
import portable_lock

LABELS=(20,21,22,23,24)
ADAPTIVE_SEED=20261025


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def emit(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def scaled_projection(projection, level):
    s=level/256
    return np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1.]]) @ projection


def read_reference(row, scenario):
    if f'reference_{scenario}_npz' in row:
        with np.load(row[f'reference_{scenario}_npz'], allow_pickle=False) as data:
            return {v:np.array(data[f'{v}_target_exact']) for v in ('AP','LAT')}
    return {v:np.load(path,allow_pickle=False) for v,path in row[f'reference_{scenario}'].items()}


def environment():
    import cma, nibabel
    return dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        torch=torch.__version__,cuda=torch.version.cuda,cma=cma.__version__,nibabel=nibabel.__version__,
        device=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory)


def configuration(args):
    base=read(REPO/'outputs/final_study/config.json')
    selection=read(REPO/'outputs/adaptive_prior_study/calibration/selection.json')
    original_base_hash=fingerprint(base)
    base={**base,'chunk_rows':args.chunk_rows}
    source_files=[Path(__file__),*sorted((REPO/'code').rglob('*.py'))]
    config=dict(version='independent_siddon_512_v2',original_base_hash=original_base_hash,
        optimization=base,adaptive=selection['adaptive'],fixed_weights=[.003,.01],starts=5,
        candidate_seed=ADAPTIVE_SEED,rotation_scale_deg=3.,translation_scale_mm=3.,
        target_renderer='exact piecewise-constant Siddon float64',target_resolution=512,
        target_downsampling='2x2 area mean, float64 accumulation then float32 target',
        siddon_chunk_rays=args.siddon_chunk_rays,registration_warm_start='new A then five new image-only candidates',
        corruption='original masks, scales and per-view RNG states; new Poisson draws for new attenuation',
        input_manifest_sha256=sha256(args.manifest),protocol_sha256=sha256(args.protocol),
        protocol_path=str(args.protocol.resolve()),input_manifest_path=str(args.manifest.resolve()),
        planned_main_scenarios=['clean']+[f'combined_s{i}' for i in range(5)],
        source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in source_files},
        environment=environment(),scope='technical pilot or frozen legacy-cohort extension; no independent-patient claim')
    out=REPO/'outputs'/args.output_name
    out.mkdir(parents=True,exist_ok=True)
    path=out/'config.json'
    if path.exists() and read(path)!=config:
        raise ValueError('Source/settings/environment changed; choose a new output-name, never overwrite frozen results')
    if not path.exists(): atomic_json(path,config)
    return config, fingerprint(config),base,out


def source_identity(row):
    paths=[Path(row['input_dir'])/n for n in ['centred_case.npz','motion_ground_truth.json']]
    paths += [Path(v) for v in row['native_paths'].values()]+[Path(row['target_metadata'])]
    if 'reference_corruption_metadata' in row: paths.append(Path(row['reference_corruption_metadata']))
    for scenario in ['clean','combined']:
        if f'reference_{scenario}_npz' in row: paths.append(Path(row[f'reference_{scenario}_npz']))
        elif f'reference_{scenario}' in row: paths += [Path(p) for p in row[f'reference_{scenario}'].values()]
    return {str(p):sha256(p) for p in dict.fromkeys(paths)}


def checked_case(name,row):
    original=read(REPO/'original_inputs_metadata/manifest.json')
    entry=next(x for x in original['cases'] if x['case']==name)
    inputs=Path(row['input_dir'])
    for filename,key in [('centred_case.npz','volume_sha256'),('motion_ground_truth.json','gt_sha256')]:
        if sha256(inputs/filename)!=entry[key]: raise ValueError(f'{name}: original cache/metadata hash mismatch')
    case=load_case(inputs,native_paths=row['native_paths'])
    meta=read(row['target_metadata'])
    if meta['case']!=name: raise ValueError('Target metadata case mismatch')
    if meta['accepted']:
        case['gt_params']=np.asarray(meta['motion']['params'])
        case['gt_matrices']=np.asarray(meta['motion']['transforms'])
    return case,meta,source_identity(row)


def save_target(directory,images,meta):
    directory.mkdir(parents=True,exist_ok=True)
    meta={**meta,'image_hashes':{v:hashlib.sha256(a.tobytes()).hexdigest() for v,a in images.items()}}
    for view,array in images.items():
        assert array.shape==(256,256) and array.dtype==np.float32 and np.isfinite(array).all()
        temp=directory/(view+'.tmp')
        with temp.open('wb') as f: np.save(f,array,allow_pickle=False)
        temp.replace(directory/(view+'.npy'))
    atomic_json(directory/'metadata.json',meta)
    return images,meta


def load_target(directory,config_hash,input_hashes):
    if not (directory/'metadata.json').exists(): return None
    meta=read(directory/'metadata.json')
    if meta['config_hash']!=config_hash or meta['source_input_hashes']!=input_hashes:
        raise ValueError('Target provenance changed')
    images={v:np.load(directory/(v+'.npy'),allow_pickle=False) for v in ('AP','LAT')}
    for v,a in images.items():
        assert a.shape==(256,256) and np.isfinite(a).all()
        assert hashlib.sha256(a.tobytes()).hexdigest()==meta['image_hashes'][v]
    return images,meta


def clean_target(name,case,legacy_meta,hashes,config,config_hash,base,out):
    directory=out/'targets/clean'/name
    cached=load_target(directory,config_hash,hashes)
    if cached is not None: return cached
    started=time.monotonic();images={};component_images={}
    geometry=ImageProblem(case,base)
    with torch.no_grad():
        for view,p in geometry.projections.items():
            parts=[]
            for i,label in enumerate(LABELS):
                part=siddon(case['components'][label],scaled_projection(p,512),case['gt_matrices'][i],
                    image_shape=(512,512),device='cuda',chunk_rays=config['siddon_chunk_rays'])
                parts.append(F.interpolate(part[None,None],size=(256,256),mode='area')[0,0])
            component_images[view]=torch.stack(parts).cpu().numpy()
            images[view]=component_images[view].sum(0).astype(np.float32)
            emit(event='target_view_done',case=name,view=view,elapsed_seconds=time.monotonic()-started)
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'components.npz',**component_images)
    meta=dict(case=name,scenario='clean',accepted=True,motion=legacy_meta['motion'],config_hash=config_hash,
        source_input_hashes=hashes,target_renderer=config['target_renderer'],target_resolution=512,
        target_downsampling=config['target_downsampling'],runtime_target_seconds=time.monotonic()-started,
        components_sha256=sha256(directory/'components.npz'))
    return save_target(directory,images,meta)


def corruption_plan(clean,parts,scales,seed,endpoint):
    """Reproduce legacy masks and noise stream positions without new-target feedback."""
    rng=np.random.default_rng(seed);plan={};replay={}
    for view in ('AP','LAT'):
        original=clean[view].astype(np.float64);scale=scales[view]
        image=parts[view][endpoint]
        support=image>.01*image.max();ys,xs=np.nonzero(support)
        assert len(xs)>=10
        x0=int(rng.integers(xs.min(),max(xs.min()+1,int(np.median(xs)))))
        choices=[]
        for width in range(1,int(xs.max()-xs.min()+2)):
            mask=np.zeros_like(support);mask[max(0,ys.min()-2):ys.max()+3,x0:x0+width]=True
            fraction=float((mask&support).sum()/support.sum())
            choices.append((abs(fraction-.15),width,fraction))
        _,width,fraction=min(choices)
        bounds=[int(max(0,ys.min()-2)),int(ys.max()+3),x0,x0+width]
        mask=np.zeros_like(support);mask[bounds[0]:bounds[1],bounds[2]:bounds[3]]=True
        state=copy.deepcopy(rng.bit_generator.state)
        transmission=np.exp(-scale*original);transmission[mask]*=.1
        observed=rng.poisson(20000*transmission)/20000
        observed+=rng.normal(0.,.003,observed.shape)
        replay[view]=(-np.log(np.clip(observed,1e-5,1.))/scale).astype(clean[view].dtype)
        plan[view]=dict(scale=float(scale),bounds=bounds,rng_state=state,original_support_coverage=fraction,
                       mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),occluded_label=LABELS[endpoint])
    return plan,replay


def corrupted_target(name,seed_index,case,row,clean,meta,config,config_hash,base,out):
    scenario=f'combined_s{seed_index}';directory=out/'targets'/scenario/name
    cached=load_target(directory,config_hash,meta['source_input_hashes'])
    if cached is not None:return cached
    started=time.monotonic();oldclean=read_reference(row,'clean')
    oldmeta=read(row.get('reference_corruption_metadata',row['target_metadata']))
    scales={v:float(oldmeta['corruption'][v]['scale']) for v in ('AP','LAT')}
    parts_path=out/'legacy_corruption_support'/name/'components.npz'
    support_manifest=parts_path.with_suffix('.json')
    if parts_path.exists():
        saved=read(support_manifest)
        if saved['config_hash']!=config_hash or saved['source_input_hashes']!=meta['source_input_hashes'] or saved['components_sha256']!=sha256(parts_path):
            raise ValueError('Original corruption support cache provenance mismatch')
        with np.load(parts_path,allow_pickle=False) as archive: parts={v:archive[v] for v in ('AP','LAT')}
    else:
        generator=ImageProblem(case,{**base,'sample_step_mm':.5,'max_samples':384})
        with torch.no_grad():
            tensors=generator.parts(torch.as_tensor(case['gt_params'],dtype=torch.float32,device='cuda'),256)
            parts={v:torch.stack(rows).cpu().numpy() for v,rows in tensors.items()}
        parts_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(parts_path,**parts)
        atomic_json(support_manifest,dict(config_hash=config_hash,source_input_hashes=meta['source_input_hashes'],components_sha256=sha256(parts_path)))
        del generator,tensors
    index=base['cases'].index(name);seed=base['seed']+index*1009+seed_index*12347
    plan,replay=corruption_plan(oldclean,parts,scales,seed,0 if (index//2)%2==0 else 4)
    replay_check={}
    if seed_index==0 and ('reference_combined_npz' in row or 'reference_combined' in row):
        expected=read_reference(row,'combined')
        for view in ('AP','LAT'):
            delta=replay[view].astype(float)-expected[view]
            replay_check[view]=dict(relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected[view]),1e-20)),
                max_abs=float(np.abs(delta).max()),bitwise_equal=bool(np.array_equal(replay[view],expected[view])))
            assert np.allclose(replay[view],expected[view],rtol=1e-6,atol=1e-6),('Original corruption replay mismatch',name,view,replay_check[view])
    images={}
    for view in ('AP','LAT'):
        p=plan[view];rng=np.random.default_rng();rng.bit_generator.state=p['rng_state']
        transmission=np.exp(-p['scale']*clean[view].astype(np.float64))
        y0,y1,x0,x1=p['bounds'];transmission[y0:y1,x0:x1]*=.1
        observed=rng.poisson(20000*transmission)/20000
        observed+=rng.normal(0.,.003,observed.shape)
        images[view]=(-np.log(np.clip(observed,1e-5,1.))/p['scale']).astype(np.float32)
    masks={}
    for view,p in plan.items():
        mask=np.zeros((256,256),dtype=bool);y0,y1,x0,x1=p['bounds'];mask[y0:y1,x0:x1]=True
        assert hashlib.sha256(mask.tobytes()).hexdigest()==p['mask_sha256']
        masks[view]=mask
    directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(directory/'masks.npz',**masks)
    return save_target(directory,images,{**meta,'scenario':scenario,'repeat':seed_index,'noise_seed':seed,
        'corruption_plan':plan,'corruption_plan_sha256':fingerprint(plan),
        'rng_state_sha256':{v:fingerprint(p['rng_state']) for v,p in plan.items()},
        'masks_archive_sha256':sha256(directory/'masks.npz'),'legacy_support_sha256':sha256(parts_path),'original_corruption_replay':replay_check,
        'runtime_corruption_seconds':time.monotonic()-started,'clean_target_hash':fingerprint(meta)})


def run_target(name,scenario,case,images,meta,config,config_hash,base,out):
    problem=ImageProblem(case,base)
    problem.set_targets({v:torch.as_tensor(a,device='cuda') for v,a in images.items()})
    result_dir=out/'results'/scenario/name;bank_dir=out/'banks'/scenario/name
    task_start=time.monotonic();timings={}
    def stage(path,method,**kwargs):
        atomic_json(out/'progress.json',dict(case=name,scenario=scenario,stage=method,status='RUNNING',pid=os.getpid()))
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();was_cached=path.exists()
        row=record(path,method,problem,case,meta,config_hash,**kwargs)
        timings[method]=dict(wall_seconds=time.monotonic()-started,cached=was_cached,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(result_dir/'timing_in_progress.json',timings)
        return row
    a=stage(result_dir/'A.json','A')
    initial=global_to_local(np.asarray(a['pose']),case['centres'])
    index=base['cases'].index(name);candidates=[]
    for k in range(5):
        start=initial if k==0 else perturb(initial,ADAPTIVE_SEED+index*1009+k*7919)
        candidates.append(stage(bank_dir/f'start_{k}.json',f'start_{k}',initial=start))
    losses=np.array([r['ncc_loss'] for r in candidates]);assert np.isfinite(losses).all()
    winner=int(losses.argmin());best=np.asarray(candidates[winner]['pose'])
    u=dispersion([x['pose'] for x in candidates],case['centres'],winner)
    atomic_json(bank_dir/'selection.json',dict(case=name,scenario=scenario,best=winner,proxy_edges=u.tolist(),
        ncc_losses=losses.tolist(),candidate_hashes=[sha256(bank_dir/f'start_{k}.json') for k in range(5)],
        target_hash=fingerprint(meta),config_hash=config_hash))
    stage(result_dir/'B.json','B',initial=best)
    for weight in [.003,.01]:stage(result_dir/f'lambda_{weight:g}.json',f'lambda_{weight:g}',initial=best,weights=weight)
    weights=config['adaptive']['lambda_max']*u/(u+config['adaptive']['tau'])
    stage(result_dir/'C_adaptive.json','C_adaptive',initial=best,weights=weights)
    rows={m:read(result_dir/(m+'.json')) for m in ['A','B','lambda_0.003','lambda_0.01','C_adaptive']}
    assert len({r['target_hash'] for r in rows.values()})==1
    assert all(np.isfinite(r['metrics']['surface_rms_tre_mm']) for r in rows.values())
    report=dict(status='COMPLETE',case=name,scenario=scenario,target_hash=fingerprint(meta),config_hash=config_hash,
        elapsed_this_invocation_seconds=time.monotonic()-task_start,stage_timings=timings,
        stage_count=10,results={m:dict(rms_tre_mm=r['metrics']['surface_rms_tre_mm'],
            centroid_failure_gt3mm=r['metrics']['case_failure_gt3mm'],runtime_seconds=r['runtime_seconds'],
            hit_max_iterations=r['optimization']['hit_max_iterations']) for m,r in rows.items()})
    atomic_json(result_dir/'task_summary.json',report)
    emit(event='task_complete',**{k:v for k,v in report.items() if k!='stage_timings'})
    del problem
    torch.cuda.empty_cache()


def phantom_checks(out):
    """Analytic voxel-box integrals, anisotropic units, transform and pixel centres."""
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    rotation=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    k=np.array([[20.,0,2],[0,20.,2],[0,0,1]])
    p=k @ np.column_stack([rotation,[0,0,10.]])
    for spacing,translation in [(np.ones(3),np.zeros(3)),(np.array([1.,2.,1.5]),np.array([1.,.7,-.6]))]:
        affine=np.eye(4);affine[:3,:3]=np.diag(spacing);affine[:3,3]=-2*spacing
        vol=Volume(np.ones((5,5,5),np.float32),spacing,affine_mm=affine)
        pose=np.eye(4);pose[:3,3]=translation
        lo=-2.5*spacing+translation;hi=2.5*spacing+translation
        source=np.linalg.solve(p[:,:3],-p[:,3]);v,u=np.mgrid[:5,:5]
        direction=np.linalg.solve(p[:,:3],np.array([u.ravel(),v.ravel(),np.ones(25)])).T
        direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(lo-source)/direction;b=(hi-source)/direction
        near=np.minimum(a,b).max(1).clip(0);far=np.maximum(a,b).min(1)
        expected=np.maximum(far-near,0).reshape(5,5)
        got=siddon(vol,p,pose,image_shape=(5,5),device='cuda',chunk_rays=7).cpu().numpy()
        np.testing.assert_allclose(got,expected,atol=1e-10,rtol=1e-12)
        reports.append(dict(spacing_mm=spacing.tolist(),translation_mm=translation.tolist(),
            max_integral_error_mm=float(np.abs(got-expected).max())))
    from spine_registration.benchmark import projections
    points=np.array([[0.,0,0],[12.,4.,8.],[-20.,-10.,15.]])
    for view,p256 in projections().items():
        uv=project_points(points,p256);uv512=project_points(points,scaled_projection(p256,512))
        np.testing.assert_allclose(uv512,2*uv+.5,atol=1e-12)
    # Every 256 pixel footprint has four symmetrically placed 512 subpixel rays.
    a=torch.arange(16,dtype=torch.float64,device='cuda').reshape(4,4)
    np.testing.assert_allclose(F.interpolate(a[None,None],size=(2,2),mode='area')[0,0].cpu(),
                              a.cpu().numpy().reshape(2,2,2,2).mean((1,3)))
    report=dict(status='PASS',analytic_box_integrals=reports,pixel_centres_and_FOV='PASS',area_average='PASS',
        scope='Small numerical/geometry checks only; no clinical validation',environment=environment())
    atomic_json(out/'phantom_checks.json',report);emit(**report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=HERE/'pilot_inputs.json')
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol/analysis_protocol.json')
    parser.add_argument('--scenarios',nargs='+',default=['clean','combined_s0'])
    parser.add_argument('--output-name',default='technical_pilot_v2')
    parser.add_argument('--chunk-rows',type=int,default=8)
    parser.add_argument('--siddon-chunk-rays',type=int,default=256)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if '/' in args.output_name or '\\' in args.output_name or '..' in args.output_name:parser.error('Simple output name required')
    if args.chunk_rows<1 or args.siddon_chunk_rays<1:parser.error('Positive chunks required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('A CUDA device is required')
    if args.self_check:
        phantom_checks(HERE/'checks');return
    lock_path=ROOT/'runner/gpu.lock'
    with lock_path.open('a+b') as lock:
        if lock_path.stat().st_size==0:lock.write(b'0');lock.flush()
        portable_lock.flock(lock,portable_lock.LOCK_EX|portable_lock.LOCK_NB)
        config,config_hash,base,out=configuration(args)
        manifest=read(args.manifest);names=args.cases or list(manifest['cases'])
        emit(event='run_start',cases=names,scenarios=args.scenarios,config_hash=config_hash,output=str(out))
        try:
            for name in names:
                if name not in base['cases']:raise ValueError('Legacy runner only accepts original 75 IDs')
                row=manifest['cases'][name];case,legacy_meta,hashes=checked_case(name,row)
                if not legacy_meta['accepted']:
                    atomic_json(out/'eligibility'/(name+'.json'),dict(accepted=False,source=row['target_metadata']))
                    continue
                clean,meta=clean_target(name,case,legacy_meta,hashes,config,config_hash,base,out)
                for scenario in args.scenarios:
                    if scenario=='clean': images,target_meta=clean,meta
                    elif scenario in [f'combined_s{i}' for i in range(5)]:
                        images,target_meta=corrupted_target(name,int(scenario[-1]),case,row,clean,meta,config,config_hash,base,out)
                    else:raise ValueError('Unknown scenario')
                    run_target(name,scenario,case,images,target_meta,config,config_hash,base,out)
            atomic_json(out/'progress.json',dict(status='COMPLETE',cases=names,scenarios=args.scenarios))
        except Exception:
            atomic_json(out/'last_error.json',dict(traceback=traceback.format_exc(),pid=os.getpid()))
            raise


if __name__=='__main__':main()
