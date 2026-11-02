"""Reproduce frozen complete-cohort intervals from the delivered target-level table.

Run in the extracted Reproducibility package: python reproduce_statistics.py
This needs NumPy, but no images, GPU, network, or confidential input locations.
"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'analysis' if (ROOT/'analysis').exists() else ROOT.parents[1]/'outputs/IJCARS_validation_revision_20261030/validation_data'
summary=json.loads((DATA/'analysis_summary.json').read_text())
with (DATA/'task_outcomes.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==2208
checks=0
for cohort,n in [('legacy',47),('external',45)]:
    ct=sorted({r['case'] for r in rows if r['cohort']==cohort});assert len(ct)==n
    methods=['B','lambda_0.003','lambda_0.01','C_adaptive']
    scenarios=['clean']+[f'combined_s{i}' for i in range(5)]
    index={(r['case'],r['scenario'],r['method']):r for r in rows if r['cohort']==cohort}
    assert len(index)==n*6*4
    x=np.array([[[[float(index[c,s,m]['rms_mm']),float(index[c,s,m]['failure_gt3mm'])] for m in methods] for s in scenarios] for c in ct])
    rng=np.random.default_rng(20261024);ix=rng.integers(0,n,size=(10000,n))
    for cond,z in [('clean',x[:,0]),('combined',x[:,1:].mean(axis=1))]:
        draws=z[ix].mean(axis=1);mean=z.mean(axis=0)
        original=summary['cohorts'][cohort]['conditions'][cond]
        for j,m in enumerate(methods):
            entry=original['methods'][m]
            np.testing.assert_allclose(mean[j],[entry['rms_mean_mm'],entry['failure_frequency']],atol=1e-13,rtol=1e-13)
            for k,field in [(0,'rms_ci95'),(1,'failure_ci95')]:
                np.testing.assert_allclose(np.quantile(draws[:,j,k],[.025,.975],method='linear'),entry[field],atol=1e-13,rtol=1e-13);checks+=1
        for r in original['contrasts']:
            j,k=methods.index(r['first']),methods.index(r['second']);q=0 if r['endpoint']=='rms_mm' else 1
            delta=z[:,j,q]-z[:,k,q];bd=draws[:,j,q]-draws[:,k,q]
            np.testing.assert_allclose(delta.mean(),r['difference'],atol=1e-13,rtol=1e-13)
            np.testing.assert_allclose(np.quantile(bd,[.025,.975],method='linear'),r['ci95'],atol=1e-13,rtol=1e-13)
            assert int(np.count_nonzero(delta))==r['discordant_cts']
            if r['primary']:np.testing.assert_allclose(np.quantile(bd,[.0125,.9875],method='linear'),r['ci975'],atol=1e-13,rtol=1e-13)
            checks+=1
print(json.dumps({'status':'PASS','target_method_rows':len(rows),'checked_intervals_and_contrasts':checks,'bootstrap_draws_per_cohort':10000,'new_tests':False}))
