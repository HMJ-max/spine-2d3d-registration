"""Outcome-blind source verification, complete image-duplicate audit, assignment.

Default: check only. --limit supports an incomplete acquisition smoke check.
Only --freeze can create an assigned manifest, and requires all 60 QC records
and at least 50 conservatively unflagged inputs. No outcomes/GPU code is read.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,itertools,json,os,shutil,sys,tempfile,zipfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
WORK=BASE.parent
sys.path[:0]=[str(WORK/'render_dependencies'),str(WORK/'render_core/src')]
import numpy as np
from spine_registration.external_dedup import compare_fingerprints

ARCHIVE=Path(r'SOURCE_LOCATION/01_current_project_core.zip')
EXPECTED_N=60
ASSIGNED_N=50
TRACKED={}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    p=Path(path)
    before=p.stat()
    with p.open('rb') as f: result=hashlib.file_digest(f,'sha256').hexdigest()
    after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'File changed during verification: {p}')
    TRACKED[str(p.resolve())]=(after.st_size,after.st_mtime_ns)
    return result

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def stable_inputs():
    for name,expected in TRACKED.items():
        p=Path(name);s=p.stat()
        require((s.st_size,s.st_mtime_ns)==expected,f'Input changed after verification: {p}')

def selection():
    protocol=read(HERE/'analysis_protocol.json')
    protocol_hash=sha(HERE/'analysis_protocol.json')
    matching=[p for p in (HERE/'freezes').glob('design_v*/receipt.json') if read(p)['sha256']['analysis_protocol.json']==protocol_hash]
    require(bool(matching),'Active analysis protocol has no immutable design freeze receipt')
    receipt_path=max(matching,key=lambda p:int(p.parent.name.split('_v')[-1]));receipt=read(receipt_path)
    order_path=BASE/'data_audit/frozen_candidate_order.json';order=read(order_path);order_hash=sha(order_path)
    require(order_hash==receipt['source_selection_sha256']['frozen_candidate_order.json'],'Authoritative candidate order changed')
    require(order_hash==protocol['cohorts']['new_public_siddon']['authoritative_candidate_order_sha256'],'Protocol/order hash mismatch')
    rows=order['ranked_candidates'];ids=[r['image_id'] for r in rows]
    require(len(ids)==len(set(ids)),'Repeated source IDs in frozen universe')
    expected=sorted(ids,key=lambda x:(hashlib.sha256((order['order_salt']+'|'+x).encode()).hexdigest(),x))
    require(ids==expected,'Frozen list does not follow its declared order')
    for r in rows:require(r['hash']==hashlib.sha256((order['order_salt']+'|'+r['image_id']).encode()).hexdigest(),'Invalid stored ordering hash')
    candidate_path=BASE/'data_audit/selected_ct_candidates.json';pool=read(candidate_path)
    require(pool['frozen_order_sha256']==order_hash,'Selected candidates use another order')
    candidates=pool['candidates']
    require(pool['target_this_run']==EXPECTED_N and len(candidates)==EXPECTED_N,'Exactly 60 selected input candidates are required')
    candidate_ids=[r['image_id'] for r in candidates]
    require(len(set(candidate_ids))==EXPECTED_N,'Repeated selected candidate')
    ranks=[int(r['rank']) for r in candidates]
    require(ranks==sorted(ranks),'Candidate ranks are not ascending')
    for r in candidates:
        require(1<=r['rank']<=len(ids) and ids[r['rank']-1]==r['image_id'],'Candidate rank/ID mismatch')
        require(r['pass'] is True,'Selected candidate failed mask prescreen')
    screening_path=BASE/'data_audit/mask_screening.json';screened=read(screening_path)
    by_rank={r['rank']:r for r in screened}
    require(len(by_rank)==len(screened),'Duplicate mask-screening rank')
    require(set(range(1,max(ranks)+1))<=set(by_rank),'Earlier source IDs lack mask-screening decisions')
    for rank in range(1,max(ranks)+1):require(by_rank[rank]['image_id']==ids[rank-1],'Screening ID/rank mismatch')
    first60=[by_rank[k]['image_id'] for k in sorted(by_rank) if by_rank[k]['pass']][:EXPECTED_N]
    require(first60==candidate_ids,'Selected 60 are not the first 60 mask-passing IDs')
    sources={'analysis_protocol.json':protocol_hash,'design_freeze_receipt_path':str(receipt_path),'design_freeze_receipt_sha256':sha(receipt_path),
             'frozen_candidate_order.json':order_hash,'selected_ct_candidates.json':sha(candidate_path),'mask_screening.json':sha(screening_path)}
    return candidates,protocol,sources

def expected_criteria():
    return {'labels':list(range(20,25)),'source_spacing_mm':1.5,'spacing_tolerance_mm':1e-5,'affine_tolerance':1e-4,
            'crop_margin_mm':3,'minimum_vertebra_voxels':16,'require_binary_disjoint_masks':True,
            'require_no_scan_boundary_contact':True,'require_descending_L1_L5_centroids':True,'centering':'mean_five_centroids',
            'near_match_handling':'quarantine pending outcome-blind source/content review; never equated to proven same patient',
            'ambiguous_pair_distance_mm':.75,'ambiguous_relative_volume_difference':.05}

def verified_fingerprint(folder,derived):
    f=read(folder/'fingerprint.json')
    with np.load(folder/'fingerprint_arrays.npz',allow_pickle=False) as data:
        require(set(data.files)=={'ct_grid','seg_grid'},'Unexpected fingerprint arrays')
        f.update({k:np.array(data[k]) for k in data.files})
    require(f['ct_sha256']==derived['ct.nii.gz'] and f['seg_sha256']==derived['seg.nii.gz'],'Fingerprint source file hash mismatch')
    require([x['label'] for x in f['label_summary']]==list(range(20,25)),'Fingerprint lacks ordered L1-L5 labels')
    compare_fingerprints(f,f) # validates finite grids and their embedded hashes
    return f

def verify_candidate(candidate,implementation):
    source_id=candidate['image_id'];raw=BASE/'data/TotalSegmentator'/source_id;folder=BASE/'external_inputs'/f'TS_{source_id}'
    manifest_path=raw/'source_manifest.json';qc_path=folder/'source_qc.json'
    result={'source_id':source_id,'case':f'TS_{source_id}','rank':candidate['rank'],'status':'PENDING'}
    if not manifest_path.exists() or not qc_path.exists():
        result['reason']='Missing completed source manifest or source QC record';return result,None
    source=read(manifest_path);qc=read(qc_path)
    require(source['image_id']==source_id and source['rank']==candidate['rank'],'Source manifest identity mismatch')
    require(source.get('ct_download_complete') is True,'CT download is not marked complete')
    require(qc['source_id']==source_id and qc['case']==f'TS_{source_id}' and qc['rank']==candidate['rank'],'QC identity mismatch')
    require(qc.get('registration_results_used') is False,'QC is not declared outcome-blind')
    require(qc['download_manifest_sha256']==sha(manifest_path),'QC points to another download manifest')
    actual_impl={str(Path(k)).replace('\\','/'):v for k,v in qc.get('qc_implementation_sha256',{}).items()}
    require(actual_impl==implementation,'QC implementation changed; rerun/version source QC')
    require(qc.get('qc_criteria')==expected_criteria(),'QC criteria differ from locked source-input criteria')
    require(qc.get('qc_criteria_sha256')==fingerprint(qc['qc_criteria']),'QC criteria hash mismatch')
    expected_masks={f'vertebrae_L{i}.nii.gz' for i in range(1,6)}
    mask_rows={Path(r['path']).name:r for r in source['files']}
    require(set(mask_rows)==expected_masks and len(source['files'])==5,'Need exactly five original mask records')
    selected_masks={Path(r['path']).name:r for r in candidate['files']}
    require(set(selected_masks)==expected_masks,'Selected candidate does not have five masks')
    raw_hashes={}
    for key,r in [*mask_rows.items(),('ct.nii.gz',source['ct'])]:
        path=Path(r['path']);expected_path=raw/key if key=='ct.nii.gz' else raw/'segmentations'/key
        require(path.resolve()==expected_path.resolve(),'Source path points outside its declared case')
        require(r['member'].endswith('/'+source_id+('/' if key=='ct.nii.gz' else '/segmentations/')+key),'Archive member/case mismatch')
        require(path.stat().st_size==int(r['bytes']),'Source file size mismatch')
        actual=sha(path);require(actual==r['sha256'],'Source SHA256 mismatch: '+key)
        if key!='ct.nii.gz':require(r['sha256']==selected_masks[key]['sha256'],'Selected/downloaded mask differs')
        raw_hashes[key]=actual
    result.update(source_manifest_sha256=sha(manifest_path),source_qc=str(qc_path),source_qc_sha256=sha(qc_path),source_hashes=raw_hashes,
                  qc_criteria_sha256=qc['qc_criteria_sha256'],qc_status=qc['status'],input_dir=str(folder),
                  native_paths={'ct':str(folder/'ct.nii.gz'),'segmentation':str(folder/'seg.nii.gz')},
                  patient_linkage_verified=False)
    require(qc['status'] in {'PASS','QUARANTINE','REJECTED_DATA_QC'},'QC did not finish with a recognized status')
    if qc['status']=='REJECTED_DATA_QC':
        result.update(status='REJECTED_DATA_QC',reason=qc.get('reason','Source QC rejection'));return result,None
    required={'ct.nii.gz','seg.nii.gz','centred_case.npz','motion_ground_truth.json','fingerprint.json','fingerprint_arrays.npz'}
    require(set(qc.get('derived_hashes',{}))==required,'Incomplete derived-file hash inventory')
    for name,digest in qc['derived_hashes'].items():require(sha(folder/name)==digest,'Derived SHA256 mismatch: '+name)
    f=verified_fingerprint(folder,qc['derived_hashes'])
    result.update(status=qc['status'],derived_hashes=qc['derived_hashes'],initial_quarantine=qc.get('flagged_matches',[]))
    return result,f

def legacy_fingerprints():
    manifest_path=BASE/'legacy_inputs/manifest_original.json';manifest=read(manifest_path)
    names=[r['case'] for r in manifest['cases']]
    require(len(names)==len(set(names))==75,'Expected 75 unique old source cases')
    result={};hashes={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in names:
            prefix=f'spine_registration_project/inputs/source_check/PPC_{name}/'
            raw_json=z.read(prefix+'fingerprint.json');raw_arrays=z.read(prefix+'fingerprint_arrays.npz')
            import io
            f=json.loads(raw_json)
            with np.load(io.BytesIO(raw_arrays),allow_pickle=False) as data:f.update({k:np.array(data[k]) for k in data.files})
            require([x['label'] for x in f['label_summary']]==list(range(20,25)),'Legacy fingerprint label mismatch')
            compare_fingerprints(f,f)
            result[name]=f;hashes[name]={'fingerprint.json':hashlib.sha256(raw_json).hexdigest(),'fingerprint_arrays.npz':hashlib.sha256(raw_arrays).hexdigest()}
    return result,{'legacy_manifest_sha256':sha(manifest_path),'archive':str(ARCHIVE),'archive_member_sha256':hashes}

def anatomy(a,b):
    aa,bb=a['label_summary'],b['label_summary']
    ca,cb=np.array([x['centroid_mm'] for x in aa]),np.array([x['centroid_mm'] for x in bb])
    va=np.array([x['voxels'] for x in aa])*abs(np.linalg.det(np.asarray(a['affine_mm'])[:3,:3]))
    vb=np.array([x['voxels'] for x in bb])*abs(np.linalg.det(np.asarray(b['affine_mm'])[:3,:3]))
    distances=[abs(np.linalg.norm(ca[i]-ca[j])-np.linalg.norm(cb[i]-cb[j])) for i,j in itertools.combinations(range(5),2)]
    distance=float(max(distances));volume=float(np.max(abs(va-vb)/np.maximum(va,vb)))
    return {'max_pair_distance_difference_mm':distance,'max_relative_volume_difference':volume,
            'anatomically_ambiguous':bool(distance<=.75 and volume<=.05)}

def compare_all(records,external,legacy):
    pairs=[];flags={r['case']:list(r.get('initial_quarantine',[])) for r in records}
    new_ids=[r['case'] for r in records]
    specifications=[(x,y,'old75_vs_new60') for x in new_ids for y in sorted(legacy)]
    specifications.extend((x,y,'new60_vs_new60') for x,y in itertools.combinations(new_ids,2))
    for left,right,kind in specifications:
        a=external.get(left);b=legacy.get(right) if right in legacy else external.get(right)
        if a is None or b is None:
            pairs.append({'left':left,'right':right,'family':kind,'status':'NOT_COMPARABLE_INPUT_QC','flagged':False,
                          'reason':'At least one candidate lacks a valid QC-approved fingerprint; that candidate is ineligible.'});continue
        row={'left':left,'right':right,'family':kind,**compare_fingerprints(a,b,ct_mse_threshold=.015,seg_mse_threshold=.015),**anatomy(a,b)}
        row['flagged']=row['status']!='distinct' or row['anatomically_ambiguous']
        if row['flagged']:
            flags[left].append({'other':right,'comparison_family':kind,'image_screen_status':row['status'],'anatomically_ambiguous':row['anatomically_ambiguous']})
            if right in external:flags[right].append({'other':left,'comparison_family':kind,'image_screen_status':row['status'],'anatomically_ambiguous':row['anatomically_ambiguous']})
        pairs.append(row)
    require(len(pairs)==len(records)*75+len(records)*(len(records)-1)//2,'Incomplete pair coverage')
    for r in records:
        r['quarantine_reasons']=flags[r['case']]
        if r['status'] in {'PASS','QUARANTINE'} and flags[r['case']]:r['status']='QUARANTINE'
    return pairs

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='Freeze50 only after complete60-case verification and full pair audit')
    parser.add_argument('--limit',type=int,help='Check only the first N candidates; never allowed with --freeze')
    parser.add_argument('--version',type=int,default=1,help='Assignment receipt version; existing version is never overwritten')
    args=parser.parse_args()
    require(not(args.freeze and args.limit is not None),'A limited check cannot freeze assignment')
    require(args.version>=1 and (args.limit is None or 1<=args.limit<=60),'Invalid version/limit')
    destination=HERE/f'freezes/external_assignment_v{args.version}'
    require(not(args.freeze and destination.exists()),'Assignment version already exists; inspect it rather than overwrite')
    candidates,protocol,sources=selection()
    implementation={p.relative_to(WORK).as_posix():sha(p) for p in [BASE/'prepare_external_inputs.py',WORK/'render_core/src/spine_registration/external_dedup.py',WORK/'render_core/src/spine_registration/benchmark.py']}
    selected=candidates[:args.limit] if args.limit is not None else candidates
    records=[];external={}
    for candidate in selected:
        try:r,f=verify_candidate(candidate,implementation)
        except (ValueError,KeyError,AssertionError,OSError) as exc:
            r={'source_id':candidate['image_id'],'case':'TS_'+candidate['image_id'],'rank':candidate['rank'],'status':'PROVENANCE_ERROR','reason':str(exc)};f=None
        records.append(r)
        if f is not None:external[r['case']]=f
    legacy,legacy_provenance=legacy_fingerprints()
    pairs=compare_all(records,external,legacy)
    eligible=[r for r in records if r['status']=='PASS']
    complete=len(records)==60 and all(r['status'] not in {'PENDING','PROVENANCE_ERROR'} for r in records)
    ready=complete and len(eligible)>=50
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'READY_TO_FREEZE' if ready else 'INCOMPLETE_OR_INSUFFICIENT_NO_FREEZE',
            'frozen':False,'limited_check':args.limit is not None,'candidate_target':60,'checked_candidates':len(records),'verified_source_qc_complete':complete,
            'input_eligible_count':len(eligible),'target_assigned':50,'assignment_shortfall':max(0,50-len(eligible)),'records':records,
            'pair_count':len(pairs),'expected_full_pair_count':6270,'pair_comparable_count':sum(p['status']!='NOT_COMPARABLE_INPUT_QC' for p in pairs),
            'flagged_pair_count':sum(p['flagged'] for p in pairs),'quarantine_policy':'Conservatively quarantine every external endpoint of an unresolved image/near/anatomical similarity flag; do not retain an arbitrary representative of an ambiguous cluster.',
            'identity_caution':'Image similarity flags and negative screens do not establish patient identity or independence. Units remain source IDs/scans.',
            'registration_or_motion_results_used':False,'protocol_and_selection_sha256':sources,'qc_implementation_sha256':implementation,
            'finalizer_sha256':sha(Path(__file__)),'legacy_provenance':legacy_provenance,'pairs':pairs}
    stable_inputs()
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit_path=HERE/'audits'/f'external_assignment_check_{timestamp}.json';write_json(audit_path,report)
    if args.freeze and ready:
        assigned=[];cases={}
        for i,r in enumerate(eligible[:50]):
            entry={**r,'case_index':i,'unit_id':r['source_id'],'unit_level':'source_id_scan','geometry_assessed':False}
            assigned.append(entry);cases[r['case']]=entry
        manifest={'version':f'external_assignment_v{args.version}','assigned_count':50,'candidate_count':60,'assigned':assigned,'cases':cases,'assigned_cases':[r['case'] for r in assigned],
                  'eligible_unassigned_reserve_ids':[r['source_id'] for r in eligible[50:]],'all_candidate_decisions':[{k:r.get(k) for k in ['source_id','case','rank','status','reason','quarantine_reasons']} for r in records],
                  'source_doi':protocol['cohorts']['new_public_siddon']['source_doi'],'original_cohort_doi':protocol['cohorts']['new_public_siddon']['original_cohort_doi'],
                  'motion_corruption_A_seed':20361024,'candidate_seed':20361025,'scenarios':['clean']+[f'combined_s{i}' for i in range(5)],
                  'geometry_exclusions_remain_assigned':True,'replacement_after_geometry_or_outcome':False,'patient_linkage_verified':False,
                  'protocol_and_selection_sha256':sources,'qc_implementation_sha256':implementation,'assignment_rule':'First50 input-eligible unflagged scans in the original frozen order after complete candidate-level QC and full pair screening.'}
        stage=Path(tempfile.mkdtemp(prefix='.assignment_staging_',dir=HERE/'freezes'))
        try:
            write_json(stage/'external_assigned_manifest.json',manifest)
            write_json(stage/'full_input_and_pair_audit.json',report)
            receipt={'freeze_id':f'external_assignment_v{args.version}','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'layer':'final_source_input_assignment','assigned_count':50,
                     'assigned_manifest_sha256':sha(stage/'external_assigned_manifest.json'),'full_audit_sha256':sha(stage/'full_input_and_pair_audit.json'),
                     'finalizer_sha256':report['finalizer_sha256'],'registration_or_motion_results_used':False,'remaining_lock':'Exact production runner/config and generated motion/targets; this assignment cannot be changed based on their outcomes.'}
            write_json(stage/'receipt.json',receipt);stable_inputs();stage.rename(destination)
        except Exception:
            # Preserve the isolated partial receipt for review; never publish it as a freeze.
            raise
        print(json.dumps({'status':'FROZEN_ASSIGNED50','directory':str(destination),'assigned_manifest_sha256':receipt['assigned_manifest_sha256'],'audit':str(audit_path)}));return 0
    print(json.dumps({k:report[k] for k in ['status','limited_check','checked_candidates','input_eligible_count','verified_source_qc_complete','assignment_shortfall','pair_count','flagged_pair_count']}|{'audit':str(audit_path),'assignment_written':False}))
    return 0 if args.limit is not None and all(r['status'] not in {'PENDING','PROVENANCE_ERROR'} for r in records) else 0 if ready and not args.freeze else 2

if __name__=='__main__':raise SystemExit(main())
