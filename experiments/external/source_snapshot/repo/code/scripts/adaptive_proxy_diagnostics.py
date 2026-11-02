"""Descriptive evaluation proxy validation and all-five-seed paired sensitivity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import adaptive_analysis as analysis
import calibrate_adaptive as calibration
import summarize_adaptive as report
from spine_registration.benchmark import atomic_json, fingerprint

ADDENDUM = 'PROXY_DIAGNOSTIC_ADDENDUM.md'
TIMING = '2026-10-24: AFTER development fixed calibration/proxy gate, BEFORE evaluation outcomes'


def diagnose_rows(rows, cases, methods, *, bootstrap_samples=10000):
    """Use complete matched main-target rows; retain no testing/gating decisions."""
    cases = list(cases)
    if len(cases) < 2 or len(set(cases)) != len(cases) or not {'B', 'C_fixed'} <= set(methods):
        raise ValueError('Require unique eligible cases and paired B/C_fixed methods')
    indexed = {(row['case'], row['scenario'], row['method']): row for row in rows}
    expected = {(case, scenario, method) for case in cases for scenario in calibration.SCENARIOS for method in methods}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('All main scenarios/methods require identical complete case coverage, without duplicates')
    comparisons = ['C_fixed'] + (['C_adaptive'] if 'C_adaptive' in methods else [])
    for case in cases:
        for scenario in calibration.SCENARIOS:
            baseline = indexed[case, scenario, 'B']
            for method in ['B', *comparisons]:
                row = indexed[case, scenario, method]
                if (row['target_hash'] != baseline['target_hash'] or row['proxy_case'] != baseline['proxy_case']
                        or row['failure_gt3mm'] not in (0, 1) or not np.isfinite(row['surface_rms_tre_mm'])
                        or row['surface_rms_tre_mm'] < 0):
                    raise ValueError('Paired target/proxy identity or endpoint validation failed')
    baseline_rows = [indexed[case, scenario, 'B'] for case in cases for scenario in calibration.SCENARIOS]
    aucs = {}
    for label, selected in [('pooled_clean_and_combined', baseline_rows),
                            ('combined_only', [row for row in baseline_rows if row['scenario'] != 'clean'])]:
        result = analysis.proxy_gate([row['proxy_case'] for row in selected],
            [row['failure_gt3mm'] for row in selected], [row['case'] for row in selected],
            bootstrap_samples=bootstrap_samples)
        result.pop('gate_passed')
        result.pop('evidence_sufficient')
        aucs[label] = dict(inference='DESCRIPTIVE_ONLY', n_rows=len(selected), **result)
    effects = []
    for scenario in calibration.SCENARIOS[1:]:
        baseline = [indexed[case, scenario, 'B'] for case in cases]
        for method in comparisons:
            first = [indexed[case, scenario, method] for case in cases]
            for endpoint, key in [('rms_mm', 'surface_rms_tre_mm'), ('failure_frequency_gt3mm', 'failure_gt3mm')]:
                result = analysis.paired_comparison([row[key] for row in first], [row[key] for row in baseline],
                                                    cases, bootstrap_samples=bootstrap_samples)
                effects.append(dict(scenario=scenario, comparison=f'{method}_minus_B', endpoint=endpoint,
                    n_volumes=result['n_volumes'], difference=result['difference'],
                    ci95_low=result['ci95'][0], ci95_high=result['ci95'][1], inference='DESCRIPTIVE_ONLY'))
    return dict(proxy_auroc=aucs, seed_effects=effects, included_cases=cases,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=analysis.BOOTSTRAP_SEED,
        adaptive_sensitivity='reported' if 'C_adaptive' in methods else 'unavailable_under_frozen_development_selection')


def write_proxy_diagnostics(output_root=calibration.OUT, *, bootstrap_samples=10000):
    root, provenance = Path(output_root), {}
    summary_path = root / 'summary/summary.json'
    if not summary_path.is_file():
        raise ValueError('Require a COMPLETE strict final summary before descriptive diagnostics')
    summary = calibration._read(summary_path, provenance)
    if summary.get('status') != 'COMPLETE':
        raise ValueError('Require a COMPLETE strict final summary before descriptive diagnostics')
    config, config_hash, split, selection, _, methods = report._identity(root, provenance)
    reporter_hash = hashlib.sha256(Path(report.__file__).read_bytes()).hexdigest()
    if (summary.get('config_hash') != config_hash or summary.get('selection') != selection
            or summary.get('methods') != methods or summary.get('reporter_sha256') != reporter_hash):
        raise ValueError('Final summary identity/selection/reporter does not match frozen inputs')
    rows, included, excluded = [], [], []
    for case in split['evaluation']:
        statuses, case_rows = [], []
        for scenario in calibration.SCENARIOS:
            status, target_rows, _ = report._target_records(root, case, scenario, config, config_hash,
                                                            selection, methods, provenance)
            statuses.append(status)
            case_rows.extend(target_rows)
        if len({status['accepted'] for status in statuses}) != 1:
            raise ValueError('Evaluation geometry eligibility differs across main targets')
        if statuses[0]['accepted']:
            included.append(case)
            rows.extend(case_rows)
        else:
            if len({fingerprint(status['motion']) for status in statuses}) != 1:
                raise ValueError('Evaluation exclusion evidence differs across main targets')
            excluded.append(case)
    coverage = summary.get('coverage', {})
    if (coverage.get('included_cases') != included or coverage.get('excluded_cases') != excluded
            or coverage.get('n_evaluation_eligible') != len(included) or coverage.get('n_assigned_evaluation') != 50):
        raise ValueError('Final summary and revalidated eligible case coverage differ')
    if any(summary.get('provenance', {}).get(path) != digest for path, digest in provenance.items()
           if path != str(summary_path)):
        raise ValueError('Final summary provenance does not match revalidated main-target inputs')
    result = diagnose_rows(rows, included, methods, bootstrap_samples=bootstrap_samples)
    addendum_path = root / ADDENDUM
    provenance[str(addendum_path)] = hashlib.sha256(addendum_path.read_bytes()).hexdigest()
    sources = [Path(__file__), Path(analysis.__file__), Path(calibration.__file__), Path(report.__file__)]
    result.update(status='COMPLETE_DESCRIPTIVE_ONLY', config_hash=config_hash,
        excluded_cases=excluded, source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        provenance=provenance, validated_rows_sha256=fingerprint(rows),
        addendum=dict(path=str(addendum_path), timing=TIMING, sha256=provenance[str(addendum_path)]),
        interpretation=dict(proxy='Evaluation predictive discrimination of maximum edge dispersion for B failure: any vertebral centroid error strictly >3 mm. Pooled and combined-only AUROC use equal CT weights and whole-CT bootstrap.',
            uncertainty='Percentile 95% CT-bootstrap intervals; proxy intervals omit draws containing only one outcome class and report their count. Null AUROC/intervals indicate unavailable class variation.',
            seed_sensitivity='All five combined seeds, identical eligible CT coverage. Negative C-minus-B effects favor C; failure effects are absolute frequency differences. No seed selection or additional hypothesis tests.',
            frozen_method='DESCRIPTIVE ONLY: never enables/disables a rule or changes lambda, tau, development gate, endpoints, or confirmatory hierarchy.',
            timing=TIMING,
            limitations='All 75 volumes were previously exposed. CT-volume independence is assumed without verified patient linkage. Development AUROC is calibration/gating evidence, not evaluation predictive validation.'))
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != provenance[str(summary_path)]:
        raise ValueError('Final summary changed during diagnostic calculation')
    destination = root / 'summary'
    report._csv(destination / 'proxy_auroc.csv', [dict(dataset=label, **value) for label, value in result['proxy_auroc'].items()])
    report._csv(destination / 'seed_effects.csv', result['seed_effects'])
    atomic_json(destination / 'proxy_diagnostics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=calibration.OUT)
    args = parser.parse_args()
    result = write_proxy_diagnostics(args.output)
    print(json.dumps(dict(status=result['status'], n_volumes=len(result['included_cases']),
                         seed_effects=len(result['seed_effects'])), sort_keys=True))


if __name__ == '__main__':
    main()
