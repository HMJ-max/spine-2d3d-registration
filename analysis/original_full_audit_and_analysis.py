"""Frozen paired, CT-level analysis; no hypothesis tests or method selection."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
INPUT=Path(__file__).resolve().parent/'downloaded/study/repo/outputs'
OUT=ROOT/'outputs/IJCARS_validation_revision_20261030'
DATA=OUT/'validation_data';DATA.mkdir(parents=True,exist_ok=True)
PROTOCOL=ROOT/'work/validation_20261026/protocol/freezes/design_v3/analysis_protocol.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fp=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
METHODS=['B','lambda_0.003','lambda_0.01','C_adaptive']
CONTRASTS=[('C_adaptive','lambda_0.01'),('lambda_0.01','B'),('C_adaptive','B'),
           ('lambda_0.003','B'),('C_adaptive','lambda_0.003'),('lambda_0.01','lambda_0.003')]
SCENARIOS=['clean']+[f'combined_s{i}' for i in range(5)]
plan=read(PROTOCOL)
assert plan['bootstrap']['seed']==20261024 and plan['bootstrap']['draws']==10000
assert plan['root_decisions']['primary_contrast']==['C_adaptive','C_lambda0.01']
summary={'protocol_sha256':sha(PROTOCOL),'bootstrap':plan['bootstrap'],
    'unit':'CT volume, patient independence unverified','cohorts_pooled':False,'cohorts':{}}
task_rows=[];volume_rows=[];contrast_rows=[];mean_rows=[];audit={}
for cohort,name,expected_n in [('legacy','legacy47_siddon_linux5090_v1',47),('external','external50_siddon_linux5090_v1',45)]:
    folder=INPUT/name;config=read(folder/'config.json');config_hash=fp(config)
    a=read(folder/'audit.json');receipt=read(folder/'run_receipt.json')
    assert a['status']=='PASS' and receipt['status']=='COMPLETE'
    assert config['protocol_sha256']==summary['protocol_sha256']
    assert a['config_hash']==config_hash
    for path,digest in config['source_hashes'].items():assert sha(folder/'source_snapshot'/path)==digest
    names=sorted(receipt['accepted_cases']);assert len(names)==expected_n and len(names)==len(set(names))
    assert len(list((folder/'results').glob('*/*/task_summary.json')))==6*expected_n
    values=np.empty((expected_n,6,4,5),dtype=float)
    checked_targets=0
    for i,case in enumerate(names):
        for s,scenario in enumerate(SCENARIOS):
            path=folder/'results'/scenario/case
            task=read(path/'task_summary.json');bank=folder/'banks'/scenario/case
            selection=read(bank/'selection.json');target=folder/'targets'/scenario/case
            meta=read(target/'metadata.json');assert meta['config_hash']==config_hash
            assert fp(meta)==task['target_hash']==selection['target_hash']
            for view in ['AP','LAT']:
                image=np.load(target/(view+'.npy'),allow_pickle=False)
                assert image.shape==(256,256) and np.isfinite(image).all()
                assert hashlib.sha256(image.tobytes()).hexdigest()==meta['image_hashes'][view]
            candidates=[read(bank/f'start_{k}.json') for k in range(5)]
            assert [sha(bank/f'start_{k}.json') for k in range(5)]==selection['candidate_hashes']
            assert selection['best']==int(np.argmin([r['ncc_loss'] for r in candidates]))
            winner=candidates[selection['best']]
            for j,method in enumerate(METHODS):
                row=read(path/(method+'.json'));metrics=row['metrics']
                assert row['config_hash']==config_hash and row['target_hash']==fp(meta)
                u=np.asarray(selection['proxy_edges']);adaptive=config['adaptive']
                weights=adaptive['lambda_max']*u/(u+adaptive['tau']) if method=='C_adaptive' else np.full(4,0.0 if method=='B' else float(method.split('_')[1]))
                np.testing.assert_array_equal(weights,row['edge_weights'])
                signature=fp(dict(method=method,case=case,weights=weights.tolist(),scales=[3.,3.],
                    initial=winner['pose'],imported_sha256=None,
                    implementation_sha256=config['source_hashes']['repo/code/scripts/adaptive_study.py']))
                assert signature==row['input_signature']
                # The recorded per-vertebra translation norm is centroid displacement.
                errors=np.array([v['translation_error_mm'] for v in row['vertebrae']])
                failures=[float(np.any(errors>t)) for t in [3,2,4,5]]
                assert bool(failures[0])==bool(metrics['case_failure_gt3mm'])
                v=[metrics['surface_rms_tre_mm'],*failures];assert np.isfinite(v).all()
                values[i,s,j]=v
                task_rows.append(dict(cohort=cohort,case=case,scenario=scenario,method=method,
                    rms_mm=v[0],failure_gt3mm=v[1],failure_gt2mm=v[2],failure_gt4mm=v[3],failure_gt5mm=v[4],
                    max_centroid_mm=float(errors.max()),runtime_seconds=row['runtime_seconds']))
            checked_targets+=1
    exclusions=[]
    if cohort=='legacy':
        exclusions=[dict(case=e['case'],category='inherited_synthetic_geometry') for e in a['cohort_coverage']['inherited_geometry_exclusions']]
    else:
        for p in sorted((folder/'eligibility').glob('*.json')):
            e=read(p)
            if not e['accepted']:
                g=e.get('motion',{}).get('geometry',{})
                exclusions.append(dict(case=e['case'],category='input_grid' if e.get('exclusion_class')=='input_geometry_not_motion' else 'synthetic_geometry',
                    max_excess_depth_mm=g.get('max_excess_depth_mm'),max_overlap_fraction=g.get('max_overlap_fraction_smaller_volume')))
    assert len(exclusions)==50-expected_n
    rng=np.random.default_rng(20261024)
    indices=rng.integers(0,expected_n,size=(10000,expected_n))
    c=dict(assigned_n=50,eligible_n=expected_n,targets=6*expected_n,corrupted_targets=5*expected_n,
        excluded=exclusions,conditions={},config_sha256=sha(folder/'config.json'),
        bootstrap_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    for condition,x in [('clean',values[:,0]),('combined',values[:,1:].mean(axis=1))]:
        means=x.mean(axis=0);draws=x[indices].mean(axis=1)
        result={'methods':{},'contrasts':[]}
        for j,method in enumerate(METHODS):
            targets=values[:,0:1,j] if condition=='clean' else values[:,1:,j]
            entry=dict(rms_mean_mm=float(means[j,0]),rms_ci95=np.quantile(draws[:,j,0],[.025,.975],method='linear').tolist(),
                failure_frequency=float(means[j,1]),failure_ci95=np.quantile(draws[:,j,1],[.025,.975],method='linear').tolist(),
                failed_targets=int(targets[:,:,1].sum()),target_count=int(targets.shape[0]*targets.shape[1]),
                ct_with_any_failure=int(np.any(targets[:,:,1]>0,axis=1).sum()),
                ct_with_any_nonfailure=int(np.any(targets[:,:,1]==0,axis=1).sum()),
                all_assigned_pipeline_success=float(expected_n/50*(1-means[j,1])),
                descriptive_failure_thresholds={str(t):float(means[j,k]) for t,k in [(2,2),(3,1),(4,3),(5,4)]})
            result['methods'][method]=entry
            mean_rows.append(dict(cohort=cohort,condition=condition,method=method,eligible_n=expected_n,**{k:v for k,v in entry.items() if not isinstance(v,(dict,list))}))
            for i,case in enumerate(names):
                volume_rows.append(dict(cohort=cohort,condition=condition,case=case,method=method,
                    rms_mm=float(x[i,j,0]),failure_frequency=float(x[i,j,1])))
        for first,second in CONTRASTS:
            j,k=METHODS.index(first),METHODS.index(second)
            differences=x[:,j,:2]-x[:,k,:2];resamples=draws[:,j,:2]-draws[:,k,:2]
            primary=condition=='combined' and (first,second)==CONTRASTS[0]
            for z,endpoint in enumerate(['rms_mm','failure_frequency']):
                entry=dict(first=first,second=second,endpoint=endpoint,primary=primary,
                    difference=float(differences[:,z].mean()),
                    ci95=np.quantile(resamples[:,z],[.025,.975],method='linear').tolist(),
                    ci975=np.quantile(resamples[:,z],[.0125,.9875],method='linear').tolist() if primary else None,
                    discordant_cts=int(np.count_nonzero(differences[:,z])),
                    empirical_zero_variance=bool(np.var(differences[:,z])==0))
                result['contrasts'].append(entry)
                contrast_rows.append(dict(cohort=cohort,condition=condition,**entry))
        c['conditions'][condition]=result
    summary['cohorts'][cohort]=c
    audit[cohort]=dict(status='PASS',source_files=len(config['source_hashes']),verified_targets=checked_targets,
        matched_final_outputs=checked_targets*4,original_audit_sha256=sha(folder/'audit.json'),
        downloaded_source_and_target_bytes_rechecked=True,matched_starts_rechecked=True)
(DATA/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
(DATA/'local_integrity_audit.json').write_text(json.dumps(audit,indent=2))
for name,rows in [('task_outcomes.csv',task_rows),('volume_outcomes.csv',volume_rows),('method_means.csv',mean_rows),('paired_contrasts.csv',contrast_rows)]:
    with (DATA/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name,c in summary['cohorts'].items():
    print(name,'n=',c['eligible_n'])
    for m,e in c['conditions']['combined']['methods'].items():print(m,round(e['rms_mean_mm'],6),round(100*e['failure_frequency'],4))
    for e in c['conditions']['combined']['contrasts']:
        if e['primary']:print('PRIMARY',e)
