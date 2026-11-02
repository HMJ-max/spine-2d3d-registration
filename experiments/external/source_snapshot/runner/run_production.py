"""Unified frozen legacy/external Siddon validation, one GPU and resumable stages."""
from pathlib import Path
import argparse, copy, json, os, sys, time, traceback
import run_siddon_validation_v2 as run
import numpy as np
import torch
from torch.nn import functional as F
from run_final_study import motion_for
from posture_geometry import prepare_geometry


def frozen_config(args,manifest):
    original=run.read(run.REPO/'outputs/final_study/config.json')
    selection=run.read(run.REPO/'outputs/adaptive_prior_study/calibration/selection.json')
    base=copy.deepcopy(original);base['chunk_rows']=args.chunk_rows
    assigned=manifest.get('assigned_cases',list(manifest['cases']))
    if args.cohort=='external':
        receipt_path=args.manifest.parent/'receipt.json'
        assignment_receipt=run.read(receipt_path)
        assert assignment_receipt['assigned_count']==manifest['assigned_count']==len(assigned)==50
        assert assignment_receipt['assigned_manifest_sha256']==run.sha256(args.manifest)
        assert len(assigned)==len(set(assigned)) and all(name in manifest['cases'] for name in assigned)
        base['cases']=assigned;base['seed']=20361024;candidate_seed=20361025
        for index,name in enumerate(assigned):assert manifest['cases'][name]['case_index']==index
    else:
        assert all(name in original['cases'] for name in manifest['cases']);candidate_seed=20261025
    run.ADAPTIVE_SEED=candidate_seed
    source_files=[Path(__file__),Path(run.__file__),*sorted((run.REPO/'code').rglob('*.py'))]
    config=dict(version='independent_siddon_production_v2',cohort=args.cohort,assigned_cases=assigned,
        eligible_input_cases=list(manifest['cases']),original_base_hash=run.fingerprint(original),optimization=base,
        adaptive=selection['adaptive'],fixed_weights=[.003,.01],starts=5,candidate_seed=candidate_seed,
        rotation_scale_deg=3.,translation_scale_mm=3.,target_renderer='exact piecewise-constant Siddon float64',
        target_resolution=512,target_downsampling='2x2 area mean, float64 accumulation then float32 target',
        siddon_chunk_rays=args.siddon_chunk_rays,planned_main_scenarios=['clean']+[f'combined_s{i}' for i in range(5)],
        corruption='same trilinear reference mechanism in both cohorts; fixed masks/scales/per-view RNG state',
        input_manifest_path=str(args.manifest.resolve()),input_manifest_sha256=run.sha256(args.manifest),
        protocol_path=str(args.protocol.resolve()),protocol_sha256=run.sha256(args.protocol),
        source_hashes={str(p.relative_to(run.ROOT)).replace('\\','/'):run.sha256(p) for p in source_files},
        environment=run.environment(),scope='Separate cohort analyses; synthetic numerical model, no clinical accuracy claim')
    if args.cohort=='external':
        config['assignment_receipt_path']=str(receipt_path.resolve())
        config['assignment_receipt_sha256']=run.sha256(receipt_path)
    out=run.REPO/'outputs'/args.output_name;out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        if run.read(out/'config.json')!=config:raise ValueError('Frozen production source/config/manifest/protocol/environment changed; new version required')
    else:run.atomic_json(out/'config.json',config)
    for path in source_files:
        destination=out/'source_snapshot'/path.relative_to(run.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert run.sha256(destination)==run.sha256(path)
        else:destination.write_bytes(path.read_bytes())
    return config,run.fingerprint(config),base,out


def external_reference(name,row,case_index,base,config_hash,out):
    """Validate frozen source bytes, generate motion once and trilinear reference only."""
    qc_path=Path(row['source_qc'])
    if run.sha256(qc_path)!=row['source_qc_sha256']:raise ValueError('Frozen external QC hash mismatch')
    qc=run.read(qc_path);assert qc['status']=='PASS' and qc['case']==name
    inputs=Path(row['input_dir'])
    if qc['derived_hashes']!=row['derived_hashes']:raise ValueError('Assigned derived hashes differ from QC')
    for filename,digest in row['derived_hashes'].items():
        if run.sha256(inputs/filename)!=digest:raise ValueError(f'{name}: derived input hash mismatch: {filename}')
    hashes={str(inputs/f):digest for f,digest in row['derived_hashes'].items()}
    hashes[str(qc_path)]=row['source_qc_sha256']
    case=run.load_case(inputs,native_paths=row['native_paths'])
    directory=out/'reference_clean'/name;geometry_path=out/'geometry'/(name+'.json')
    if geometry_path.exists():
        motion_meta=run.read(geometry_path)
        assert motion_meta['config_hash']==config_hash and motion_meta['source_input_hashes']==hashes
    else:
        with np.load(inputs/'centred_case.npz',allow_pickle=False) as arrays:
            prepared=prepare_geometry(arrays['segmentation'],arrays['affine_mm'])
        try:
            motion=motion_for(case,prepared,case_index,base,'moderate')
            motion_meta=dict(case=name,case_index=case_index,accepted=bool(motion['accepted']),motion=motion,
                             config_hash=config_hash,source_input_hashes=hashes)
        except ValueError as exc:
            if 'anatomy outside detector safety margin' not in str(exc):raise
            motion_meta=dict(case=name,case_index=case_index,accepted=False,reason='initial_detector_field_of_view',
                             detail=str(exc),config_hash=config_hash,source_input_hashes=hashes)
        from generate_posture_dataset import plain
        run.atomic_json(geometry_path,plain(motion_meta));motion_meta=run.read(geometry_path)
    if not motion_meta['accepted']:return case,None,None,motion_meta
    case['gt_params']=np.asarray(motion_meta['motion']['params'])
    case['gt_matrices']=np.asarray(motion_meta['motion']['transforms'])
    cached=run.load_target(directory,config_hash,hashes)
    if cached is None:
        start=time.monotonic();generator=run.ImageProblem(case,{**base,'sample_step_mm':.5,'max_samples':384})
        with torch.no_grad():
            parts=generator.parts(torch.as_tensor(case['gt_params'],dtype=torch.float32,device='cuda'),512)
            arrays={v:F.interpolate(torch.stack(rows).sum(0)[None,None],size=(256,256),mode='area')[0,0].cpu().numpy()
                    for v,rows in parts.items()}
        scales={v:float(np.log(20.)/np.quantile(x[x>0],.995)) for v,x in arrays.items()}
        reference_meta={**motion_meta,'scenario':'trilinear_reference_only','corruption':{v:{'scale':s} for v,s in scales.items()},
                        'runtime_reference_seconds':time.monotonic()-start,'registration_performed_on_reference':False}
        run.save_target(directory,arrays,reference_meta)
        del generator,parts
    reference_row={**row,'target_metadata':str(directory/'metadata.json'),
        'reference_corruption_metadata':str(directory/'metadata.json'),
        'reference_clean':{v:str(directory/(v+'.npy')) for v in ['AP','LAT']}}
    target_meta=run.read(directory/'metadata.json')
    source_hashes=run.source_identity(reference_row);source_hashes[str(qc_path)]=row['source_qc_sha256']
    return case,reference_row,source_hashes,target_meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['legacy','external'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output-name',required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--scenarios',nargs='+',default=['clean']+[f'combined_s{i}' for i in range(5)])
    parser.add_argument('--chunk-rows',type=int,default=32)
    parser.add_argument('--siddon-chunk-rays',type=int,default=256)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if any(x in args.output_name for x in ['/','\\','..']):parser.error('Simple output name required')
    if any(s not in ['clean']+[f'combined_s{i}' for i in range(5)] for s in args.scenarios):parser.error('Unknown scenario')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with (run.HERE/'gpu.lock').open('a+b') as lock:
        run.portable_lock.flock(lock,run.portable_lock.LOCK_EX|run.portable_lock.LOCK_NB)
        manifest=run.read(args.manifest);config,config_hash,base,out=frozen_config(args,manifest)
        names=args.cases or list(manifest['cases']);accepted=[];excluded=[];prepared_targets=[]
        start=time.monotonic()
        receipt=dict(status='RUNNING',cohort=args.cohort,cases=names,scenarios=args.scenarios,prepare_only=args.prepare_only,
            config_hash=config_hash,manifest_sha256=config['input_manifest_sha256'],protocol_sha256=config['protocol_sha256'],
            source_hashes=config['source_hashes'],pid=os.getpid())
        run.atomic_json(out/'run_receipt.json',receipt);run.emit(event='run_start',**{k:v for k,v in receipt.items() if k!='source_hashes'})
        try:
            for name in names:
                if name not in manifest['cases']:raise ValueError('Case absent from frozen manifest')
                row=manifest['cases'][name]
                if args.cohort=='legacy':case,meta,hashes=run.checked_case(name,row)
                else:case,row,hashes,meta=external_reference(name,row,base['cases'].index(name),base,config_hash,out)
                if not meta['accepted']:
                    excluded.append(name);run.atomic_json(out/'eligibility'/(name+'.json'),meta);continue
                accepted.append(name);run.atomic_json(out/'eligibility'/(name+'.json'),dict(case=name,accepted=True,config_hash=config_hash))
                clean,clean_meta=run.clean_target(name,case,meta,hashes,config,config_hash,base,out)
                for scenario in args.scenarios:
                    if scenario=='clean':images,target_meta=clean,clean_meta
                    else:images,target_meta=run.corrupted_target(name,int(scenario[-1]),case,row,clean,clean_meta,config,config_hash,base,out)
                    prepared_targets.append({'case':name,'scenario':scenario,'target_hash':run.fingerprint(target_meta)})
                    if not args.prepare_only:run.run_target(name,scenario,case,images,target_meta,config,config_hash,base,out)
                del case;torch.cuda.empty_cache()
            if not args.prepare_only:
                for name in accepted:
                    for scenario in args.scenarios:
                        summary=run.read(out/'results'/scenario/name/'task_summary.json')
                        assert summary['status']=='COMPLETE' and summary['config_hash']==config_hash
            complete_all=not args.prepare_only and set(names)==set(manifest['cases']) and set(args.scenarios)==set(config['planned_main_scenarios'])
            receipt.update(status='TARGETS_PREPARED' if args.prepare_only else 'COMPLETE' if complete_all else 'SUBSET_COMPLETE',
                complete_all_assigned_requested_tasks=complete_all,accepted_cases=accepted,geometry_excluded_cases=excluded,
                prepared_target_count=len(prepared_targets),prepared_targets=prepared_targets,
                completed_target_count=0 if args.prepare_only else len(prepared_targets),elapsed_seconds=time.monotonic()-start)
            run.atomic_json(out/'run_receipt.json',receipt);run.atomic_json(out/'progress.json',receipt)
            run.emit(event='run_complete',**{k:v for k,v in receipt.items() if k not in ['source_hashes','prepared_targets']})
        except Exception:
            receipt.update(status='ERROR',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
            run.atomic_json(out/'last_error.json',receipt);run.atomic_json(out/'run_receipt.json',receipt)
            raise


if __name__=='__main__':
    try:
        main()
    except Exception:
        run.atomic_json(run.HERE/'production_invocation_error.json',dict(status='ERROR',argv=sys.argv,
            pid=os.getpid(),traceback=traceback.format_exc()))
        raise
