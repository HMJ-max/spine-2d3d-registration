"""Strict evaluation-only reporting after development selection is locked."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_analysis import LAMBDA_GRID, FAILURE_THRESHOLDS_MM, joint_superiority, summarize_outcomes
from adaptive_study import relative_frames
from calibrate_adaptive import ROOT, OUT, SCENARIOS, _read, _record
from spine_registration.benchmark import atomic_json, fingerprint, global_to_local
from spine_registration.torch_registration import se3_log


METRICS = ('surface_rms_tre_mm', 'translation_error_mm', 'rotation_error_deg',
           'relative_translation_error_mm', 'relative_rotation_error_deg')
OBJECTIVES = ('ncc_loss', 'motion_penalty', 'weighted_prior', 'weighted_fraction')


def _identity(root, provenance):
    config = _read(root / 'config.json', provenance)
    split = _read(root / 'split.json', provenance)
    config_hash = fingerprint(config)
    if split != config['split'] or tuple(config['main_scenarios']) != SCENARIOS or tuple(config['lambda_grid']) != LAMBDA_GRID:
        raise ValueError('Frozen configuration/split/grid mismatch')
    if (len(split['evaluation']) != 50 or len(set(split['evaluation'])) != 50
            or len(split['development']) != 25 or set(split['evaluation']) & set(split['development'])):
        raise ValueError('Expected a disjoint 25/50 volume split')
    for relative, digest in config['source_hashes'].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    selection = _read(root / 'calibration/selection.json', provenance)
    fixed = _read(root / 'calibration/fixed_selection.json', provenance)
    gate = _read(root / 'calibration/proxy_gate.json', provenance)
    if (any(x.get('config_hash') != config_hash for x in (selection, fixed, gate))
            or selection.get('fixed_selection_hash') != fingerprint(fixed)
            or selection.get('proxy_gate_hash') != fingerprint(gate)
            or selection.get('fixed_lambda') != fixed.get('fixed_lambda')
            or selection['fixed_lambda'] not in LAMBDA_GRID):
        raise ValueError('Development selection/configuration hashes do not match')
    adaptive = selection.get('adaptive')
    if bool(adaptive is not None) != bool(gate['gate_passed']):
        raise ValueError('A passed proxy gate requires completed adaptive calibration; a failed gate forbids adaptive evaluation')
    if adaptive is not None and (adaptive.get('lambda_max') not in LAMBDA_GRID
            or not np.isfinite(adaptive.get('tau', np.nan)) or adaptive['tau'] <= 0):
        raise ValueError('Invalid frozen adaptive parameters')
    methods = {'Zero': 'Zero', 'A': 'A', 'B': 'B',
               'C_fixed': 'B' if selection['fixed_lambda'] == 0 else f"lambda_{selection['fixed_lambda']:g}",
               'C_lambda0.1': 'lambda_0.1'}
    if adaptive is not None:
        methods['C_adaptive'] = 'C_adaptive'
    return config, config_hash, split, selection, gate, methods


def _validate_metrics(record):
    for metric in METRICS:
        values = [v[metric] for v in record['relative_edges' if metric.startswith('relative_') else 'vertebrae']]
        expected = 4 if metric.startswith('relative_') else 5
        if (len(values) != expected or not np.isfinite(values).all() or min(values) < 0
                or not np.isclose(record['metrics'][metric], np.mean(values), rtol=1e-8, atol=1e-10)):
            raise ValueError(f'Invalid metric aggregation: {record["case"]}/{record["method"]}/{metric}')


def _motion_recovery(record, metadata):
    centres = torch.tensor(metadata['centres'], dtype=torch.float64)
    truth = torch.tensor(metadata['motion']['transforms'], dtype=torch.float64)
    if centres.shape != (5, 3) or truth.shape != (5, 4, 4) or not torch.isfinite(centres).all() or not torch.isfinite(truth).all():
        raise ValueError('Invalid ground-truth motion metadata')
    offsets = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    offsets[:, :3, 3] = centres
    pre = torch.linalg.solve(offsets[:-1], offsets[1:])
    gt_frames = truth @ offsets
    gt_relative = torch.linalg.solve(gt_frames[:-1], gt_frames[1:])
    pose = np.asarray(record['pose'])
    if record['method'] == 'A':
        pose = global_to_local(pose, centres.numpy())
    estimate_relative = relative_frames(torch.tensor(pose, dtype=torch.float64), centres)
    gt = se3_log(torch.linalg.solve(pre, gt_relative)).numpy()
    estimate = se3_log(torch.linalg.solve(pre, estimate_relative)).numpy()
    gt_rot, est_rot = np.rad2deg(gt[:, :3]), np.rad2deg(estimate[:, :3])
    result = dict(adjacent_gt_rotation_deg=float(np.linalg.norm(gt_rot, axis=1).mean()),
        adjacent_estimated_rotation_deg=float(np.linalg.norm(est_rot, axis=1).mean()),
        adjacent_rotation_vector_rmse_deg=float(np.sqrt(np.mean(np.sum((est_rot - gt_rot) ** 2, axis=1)))),
        adjacent_gt_translation_mm=float(np.linalg.norm(gt[:, 3:], axis=1).mean()),
        adjacent_estimated_translation_mm=float(np.linalg.norm(estimate[:, 3:], axis=1).mean()),
        adjacent_translation_vector_rmse_mm=float(np.sqrt(np.mean(np.sum((estimate[:, 3:] - gt[:, 3:]) ** 2, axis=1)))))
    for edge in range(4):
        result[f'edge{edge}_gt_sagittal_deg'] = float(gt_rot[edge, 0])
        result[f'edge{edge}_estimated_sagittal_deg'] = float(est_rot[edge, 0])
    return result


def _objective(record):
    values = {key: float(record[key]) for key in OBJECTIVES}
    penalties = np.asarray(record['edge_penalties'], float)
    weighted = float(np.mean(penalties * np.asarray(record['edge_weights'])))
    if (penalties.shape != (4,) or not np.isfinite(penalties).all() or np.any(penalties < 0)
            or not np.isfinite(list(values.values())).all()
            or not np.isclose(values['motion_penalty'], penalties.mean())
            or not np.isclose(values['weighted_prior'], weighted)
            or not np.isclose(values['weighted_fraction'], weighted / max(values['ncc_loss'] + weighted, 1e-20))):
        raise ValueError('Objective/penalty aggregation mismatch')
    return values


def _runtime(record):
    value = record.get('imported_provenance', {}).get('original_runtime_seconds', record['runtime_seconds'])
    if not np.isfinite(value) or value < 0:
        raise ValueError('Invalid recorded runtime')
    return float(value)


def _target_records(root, case, scenario, config, config_hash, selection, methods, provenance, manifest=None, scales_by_method=None):
    eligibility = _read(root / 'eligibility' / scenario / f'{case}.json', provenance)
    if eligibility.get('case') != case or type(eligibility.get('accepted')) is not bool:
        raise ValueError('Invalid evaluation eligibility identity/status')
    if not eligibility['accepted']:
        if not isinstance(eligibility.get('motion'), dict) or eligibility['motion'].get('accepted') is not False:
            raise ValueError('Missing explicit geometry rejection evidence')
        source = Path(eligibility['source'])
        if not (source / 'metadata.json').is_file():
            baseline = _read(root / 'eligibility/clean' / f'{case}.json', provenance)
            source = Path(baseline['source'])
        rejected = _read(source / 'metadata.json', provenance)
        if (source.name != case or rejected.get('case') != case or rejected.get('accepted') is not False
                or rejected.get('config_hash') != config['base_config_hash']
                or fingerprint(rejected.get('motion')) != fingerprint(eligibility['motion'])):
            raise ValueError('Geometry rejection metadata/provenance mismatch')
        if manifest is not None and rejected.get('robustness_manifest_hash') != fingerprint(manifest):
            raise ValueError('Supplementary rejection manifest mismatch')
        return eligibility, [], []
    source = Path(eligibility['source'])
    if source.name != case:
        raise ValueError('Target source belongs to a different volume')
    metadata = _read(source / 'metadata.json', provenance)
    expected_scenario = {'clean': 'moderate', 'combined_s0': 'combined'}.get(scenario, scenario)
    if (metadata.get('case') != case or metadata.get('scenario') != expected_scenario
            or metadata.get('config_hash') != config['base_config_hash'] or metadata.get('accepted') is not True
            or fingerprint(metadata) != eligibility.get('target_hash')):
        raise ValueError('Evaluation target metadata/hash mismatch')
    if manifest is not None and metadata.get('robustness_manifest_hash') != fingerprint(manifest):
        raise ValueError('Supplementary target manifest mismatch')
    directory, bank = root / 'results' / scenario / case, root / 'banks' / scenario / case
    runner_hash = config['source_hashes']['code/scripts/adaptive_study.py']
    records = {method: _record(directory / f'{method}.json', method, case, scenario, eligibility['target_hash'],
                              config_hash, runner_hash, provenance) for method in set(methods.values())}
    candidates = [_record(bank / f'start_{k}.json', f'start_{k}', case, scenario, eligibility['target_hash'],
                          config_hash, runner_hash, provenance) for k in range(5)]
    proxy = _read(bank / 'selection.json', provenance)
    losses, edges = [x['ncc_loss'] for x in candidates], np.asarray(proxy['proxy_edges'], float)
    if (any(proxy.get(k) != v for k, v in dict(case=case, scenario=scenario, config_hash=config_hash,
            target_hash=eligibility['target_hash']).items()) or proxy['best'] != int(np.argmin(losses))
            or not np.array_equal(proxy['ncc_losses'], losses) or edges.shape != (4,)
            or not np.isfinite(edges).all() or np.any(edges < 0) or proxy['proxy_case'] != float(edges.max())):
        raise ValueError('Invalid evaluation candidate selection/proxy')
    bank_cost = sum(_runtime(x) for x in candidates)
    if not np.isclose(proxy['candidate_runtime_seconds'], bank_cost, rtol=1e-8, atol=1e-8):
        raise ValueError('Candidate-bank runtime aggregation mismatch')
    a_cost, rows, convergence = _runtime(records['A']), [], []
    for display, source_method in methods.items():
        record = records[source_method]
        _validate_metrics(record)
        expected_weights = np.zeros(4)
        if source_method.startswith('lambda_'):
            expected_weights[:] = float(source_method[7:])
        elif source_method.startswith('C_scale_'):
            expected_weights[:] = selection['fixed_lambda']
        elif source_method == 'C_adaptive':
            rule = selection['adaptive']
            expected_weights = rule['lambda_max'] * edges / (edges + rule['tau'])
        expected_scales = (scales_by_method or {}).get(display, [3., 3.])
        if record.get('scales') != expected_scales or not np.allclose(record['edge_weights'], expected_weights, rtol=1e-10, atol=1e-12):
            raise ValueError('Evaluation weights differ from frozen selection')
        cost = _runtime(record)
        accounted = cost if display in ('A', 'Zero') else a_cost + bank_cost + cost
        row = dict(case=case, scenario=scenario, method=display, source_method=source_method,
            target_hash=eligibility['target_hash'], edge_weights=record['edge_weights'],
            edge_penalties=record['edge_penalties'], prior_scales=record['scales'],
            **{k: record['metrics'][k] for k in METRICS},
            **_objective(record), **_motion_recovery(record, metadata), failure_gt3mm=int(record['metrics']['case_failure_gt3mm']),
            runtime_refinement_seconds=cost, runtime_A_seconds=a_cost if display not in ('A', 'Zero') else 0.,
            runtime_bank_seconds=bank_cost if display not in ('A', 'Zero') else 0.,
            runtime_accounted_seconds=accounted, proxy_runtime_seconds=None, proxy_case=proxy['proxy_case'])
        errors = [v['translation_error_mm'] for v in record['vertebrae']]
        for threshold in FAILURE_THRESHOLDS_MM:
            row[f'failure_gt{threshold:g}mm'] = int(max(errors) > threshold)
        row['centroid_errors'] = errors
        rows.append(row)
        diag = record['optimization']
        convergence.append(dict(case=case, scenario=scenario, method=display, source_method=source_method,
            stop_reason=diag['stop_reason'], converged=diag['converged'], hit_max_iterations=diag['hit_max_iterations'],
            iterations=diag['iterations'], evaluations=diag['evaluations'], ncc_loss=record['ncc_loss'],
            runtime_accounted_seconds=accounted, proxy_runtime_seconds=None))
    return eligibility, rows, convergence


def _describe(rows, assigned, label, bootstrap_samples):
    summaries, volumes, thresholds = {}, [], []
    for method in sorted({x['method'] for x in rows}):
        selected = [x for x in rows if x['method'] == method]
        ids = [x['case'] for x in selected]
        result = summarize_outcomes([x['surface_rms_tre_mm'] for x in selected],
            [x['centroid_errors'] for x in selected], ids, bootstrap_samples=bootstrap_samples)
        numeric = [key for key, value in selected[0].items() if isinstance(value, (int, float))]
        local_volumes = []
        for case in sorted(set(ids)):
            case_rows = [x for x in selected if x['case'] == case]
            volume = dict(dataset=label, case=case, method=method, source_method=case_rows[0]['source_method'],
                n_targets=len(case_rows), **{key: float(np.mean([x[key] for x in case_rows])) for key in numeric})
            local_volumes.append(volume)
        volumes.extend(local_volumes)
        result['additional_volume_mean_metrics'] = {
            key: dict(mean=float(np.mean([x[key] for x in local_volumes])),
                      median=float(np.median([x[key] for x in local_volumes])),
                      p95=float(np.quantile([x[key] for x in local_volumes], .95)))
            for key in numeric if key != 'surface_rms_tre_mm'}
        values, counts = np.unique(result['volume_rms'], return_counts=True)
        result['volume_rms_ecdf'] = dict(rms_mm=values.tolist(), cumulative_fraction=(counts.cumsum() / counts.sum()).tolist())
        result['n_assigned'] = assigned
        result['geometry_coverage'] = len(local_volumes) / assigned
        result['pipeline_success_frequency_all_assigned_gt3mm'] = sum(1 - x['failure_gt3mm'] for x in local_volumes) / assigned
        result['proxy_runtime_seconds'] = None
        result['runtime_interpretation'] = 'Accounted serial component cost includes original A and all five bank costs for B/C; unmeasured proxy overhead omitted, not end-to-end clinical latency.'
        summaries[method] = result
        for threshold, failure in result['failures_by_threshold_mm'].items():
            thresholds.append(dict(dataset=label, method=method, threshold_mm=float(threshold),
                n_eligible=result['n_volumes'], n_assigned=assigned,
                mean_failure_frequency=failure['mean_frequency'], ci95_low=failure['ci95'][0], ci95_high=failure['ci95'][1],
                pipeline_success_frequency_all_assigned=sum(1 - x[f'failure_gt{threshold}mm'] for x in local_volumes) / assigned))
    return summaries, volumes, thresholds


def _comparisons(combined, gate, bootstrap_samples):
    methods = {x['method'] for x in combined}
    pairs = [('C_fixed', 'B', 'main')]
    if 'C_adaptive' in methods:
        pairs += [('C_adaptive', 'B', 'adaptive_extension'), ('C_adaptive', 'C_fixed', 'key_secondary')]
    results, differences, previous_rejected = [], [], True
    for first, second, role in pairs:
        a = sorted((x for x in combined if x['method'] == first), key=lambda x: (x['case'], x['scenario']))
        b = sorted((x for x in combined if x['method'] == second), key=lambda x: (x['case'], x['scenario']))
        if [(x['case'], x['scenario'], x['target_hash']) for x in a] != [(x['case'], x['scenario'], x['target_hash']) for x in b]:
            raise ValueError('Paired inference targets do not match')
        result = joint_superiority([x['surface_rms_tre_mm'] for x in a], [x['surface_rms_tre_mm'] for x in b],
            [x['failure_gt3mm'] for x in a], [x['failure_gt3mm'] for x in b], [x['case'] for x in a],
            bootstrap_samples=bootstrap_samples)
        open_gate = previous_rejected and (role == 'main' or gate['gate_passed'])
        confirmed = open_gate and result['joint_superiority']
        results.append(dict(first=first, second=second, role=role,
            inference_status='confirmatory' if open_gate else 'exploratory_gate_closed',
            confirmatory_joint_superiority=bool(confirmed), **result))
        previous_rejected = confirmed
        for endpoint in ('rms', 'failure_frequency'):
            for case, delta in zip(result[endpoint]['volume_ids'], result[endpoint]['volume_differences']):
                differences.append(dict(comparison=f'{first}_minus_{second}', case=case, endpoint=endpoint, difference=delta))
    return results, differences


def _optional_missing(root, names, scenario, methods):
    missing = []
    for case in names:
        path = root / 'eligibility' / scenario / f'{case}.json'
        if not path.is_file():
            missing.append(str(path)); continue
        eligibility = json.loads(path.read_text())
        if eligibility.get('accepted') is False:
            continue
        paths = [root / 'results' / scenario / case / f'{method}.json' for method in set(methods.values())]
        paths += [root / 'banks' / scenario / case / f'start_{k}.json' for k in range(5)]
        paths += [root / 'banks' / scenario / case / 'selection.json']
        missing.extend(str(p) for p in paths if not p.is_file())
    return missing


def _optional(root, config, config_hash, split, selection, methods, provenance, bootstrap_samples):
    datasets, rows, convergence, volumes, thresholds = {}, [], [], [], []
    tasks = []
    path = root / 'robustness/manifest.json'
    if path.is_file():
        manifest = _read(path, provenance)
        if (manifest['config_hash'] != config_hash or manifest['cases'] != split['evaluation'][:10]
                or manifest['selection_sha256'] != provenance[str(root / 'calibration/selection.json')]
                or manifest['split_sha256'] != provenance[str(root / 'split.json')]
                or manifest['selected_lambda'] != selection['fixed_lambda']):
            raise ValueError('Supplementary manifest/selection/split mismatch')
        for relative, digest in manifest['source_hashes'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Supplementary implementation changed')
        scenarios = manifest['motion_scenarios'] + list(manifest['corruption_scenarios'])
        for scenario in scenarios:
            tasks.append((scenario, scenario, manifest['cases'], methods, manifest, None))
        scales = {f'C_scale_t{x[1]:g}': x for x in manifest['prior_scales']}
        for scenario in manifest['prior_scale_scenarios']:
            tasks.append((f'scales_{scenario}', scenario, manifest['cases'], {**methods, **{key: key for key in scales}}, None, scales))
    if any(root.glob('sensitivity_*_completion.json')) or (root / 'eligibility/siddon_clean').exists():
        for scenario in ('siddon_clean', 'siddon_combined'):
            tasks.append((scenario, scenario, split['evaluation'][:10], methods, None, None))
    for label, scenario, names, selected_methods, manifest, scales in tasks:
        missing = _optional_missing(root, names, scenario, selected_methods)
        if missing:
            datasets[label] = dict(status='INCOMPLETE_NOT_ANALYZED', n_assigned=len(names), missing=missing)
            continue
        selected, stops, excluded = [], [], []
        for case in names:
            eligibility, target_rows, target_stops = _target_records(root, case, scenario, config, config_hash,
                selection, selected_methods, provenance, manifest, scales)
            if not eligibility['accepted']:
                excluded.append(case)
            selected.extend(target_rows); stops.extend(target_stops)
        if not selected:
            datasets[label] = dict(status='NO_ELIGIBLE_TARGETS', n_assigned=len(names), excluded_cases=excluded,
                                   geometry_coverage=0, pipeline_success_frequency_all_assigned=0)
            continue
        result, local_volumes, local_thresholds = _describe(selected, len(names), label, bootstrap_samples)
        datasets[label] = dict(status='COMPLETE_DESCRIPTIVE', n_assigned=len(names), excluded_cases=excluded, methods=result,
            scale_effect='inactive when selected lambda is zero' if scales is not None else None,
            inference='descriptive; no additional confirmatory tests')
        rows.extend(dict(dataset=label, **row) for row in selected)
        convergence.extend(dict(dataset=label, **row) for row in stops)
        volumes.extend(local_volumes); thresholds.extend(local_thresholds)
    return datasets, rows, convergence, volumes, thresholds


def development_objective_summary(root):
    """Separate descriptive λ justification; never changes selected parameters."""
    from calibrate_adaptive import load_development
    data = load_development(root)
    summaries, rows = {}, []
    for case in data['included']:
        for scenario in SCENARIOS:
            for weight in LAMBDA_GRID:
                method = 'B' if weight == 0 else f'lambda_{weight:g}'
                path = root / 'results' / scenario / case / f'{method}.json'
                record = json.loads(path.read_text())
                rows.append(dict(case=case, scenario=scenario, method=method, lambda_value=weight, **_objective(record)))
    for condition in ('clean', 'combined'):
        summaries[condition] = {}
        for weight in LAMBDA_GRID:
            selected = [x for x in rows if x['lambda_value'] == weight and (x['scenario'] == 'clean') == (condition == 'clean')]
            volume_values = [{key: np.mean([x[key] for x in selected if x['case'] == case]) for key in OBJECTIVES}
                             for case in data['included']]
            summaries[condition][str(weight)] = {key: dict(mean=float(np.mean([x[key] for x in volume_values])),
                median=float(np.median([x[key] for x in volume_values])),
                p95=float(np.quantile([x[key] for x in volume_values], .95))) for key in OBJECTIVES}
    return dict(status='DESCRIPTIVE_DEVELOPMENT_OBJECTIVES', config_hash=data['config_hash'],
        included_cases=data['included'], excluded_cases=data['excluded'], summaries=summaries, rows=rows,
        interpretation='Objective magnitudes describe numerical tradeoffs, not physiological calibration. No lambda reselection.')


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def summarize(output_root=OUT, *, bootstrap_samples=10000, include_development=True):
    root, provenance = Path(output_root), {}
    config, config_hash, split, selection, gate, methods = _identity(root, provenance)
    rows, convergence, included, excluded = [], [], [], []
    for case in split['evaluation']:
        case_rows, case_stops, eligibility = [], [], []
        for scenario in SCENARIOS:
            status, target_rows, stops = _target_records(root, case, scenario, config, config_hash,
                                                        selection, methods, provenance)
            eligibility.append(status)
            case_rows.extend(target_rows); case_stops.extend(stops)
        if len({x['accepted'] for x in eligibility}) != 1:
            raise ValueError('Evaluation geometry eligibility differs across the six main targets')
        if not eligibility[0]['accepted']:
            if len({fingerprint(x['motion']) for x in eligibility}) != 1:
                raise ValueError('Evaluation exclusion evidence differs across targets')
            excluded.append(case)
        else:
            included.append(case)
            rows.extend(case_rows); convergence.extend(case_stops)
    if len(included) < 2:
        raise ValueError('At least two eligible evaluation volumes are needed for inference')
    clean = [x for x in rows if x['scenario'] == 'clean']
    combined = [x for x in rows if x['scenario'] != 'clean']
    main, volumes, thresholds = {}, [], []
    for label, targets in [('clean', clean), ('combined', combined)]:
        result, local_volumes, local_thresholds = _describe(targets, 50, label, bootstrap_samples)
        main[label] = result
        volumes.extend(local_volumes); thresholds.extend(local_thresholds)
    tests, differences = _comparisons(combined, gate, bootstrap_samples)
    optional, extra_rows, extra_stops, extra_volumes, extra_thresholds = _optional(
        root, config, config_hash, split, selection, methods, provenance, bootstrap_samples)
    all_rows = [dict(dataset='main', **row) for row in rows] + extra_rows
    all_stops = [dict(dataset='main', **row) for row in convergence] + extra_stops
    volumes.extend(extra_volumes); thresholds.extend(extra_thresholds)
    development = development_objective_summary(root) if include_development else None
    result = dict(status='COMPLETE', config_hash=config_hash, selection=selection,
        coverage=dict(n_assigned_cohort=75, n_assigned_development=25, n_assigned_evaluation=50,
                      n_evaluation_eligible=len(included), included_cases=included, excluded_cases=excluded),
        methods=methods, main=main, comparisons=tests, optional_sensitivity=optional,
        interpretation=dict(primary='C_fixed minus B; joint RMS and failure-frequency superiority in combined targets',
            independent_unit='CT volume; five combined seeds averaged within volume',
            method_aliases={key: value for key, value in methods.items() if key != value},
            A='Structural reference only', Zero='Zero-motion context only',
            lambda_tuning='Smallest lambda within one SE of minimum risk: nonstandard motion-preserving tolerance heuristic; classical glmnet one-SE favors larger lambda.',
            proxy_runtime_seconds=None, proxy_runtime_status='Not separately measured; do not infer from missing timing',
            target_integrity_scope='Metadata fingerprints and record target hashes checked; target arrays are not reread by this report.',
            data_exposure='All 75 volumes previously exposed; subsequent 25/50 development/evaluation split.',
            patient_linkage=config.get('patient_linkage', 'unverified'),
            geometry_exclusions='Counted as failures for all-assigned pipeline-success frequency; no invented RMS.',
            clinical_scope='Synthetic target study; thresholds, objective scales and motion proxies do not establish clinical accuracy or safety.'),
        development_objectives_status='reported_separately' if development is not None else 'not_requested',
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), provenance=provenance)
    destination = root / 'summary'
    destination.mkdir(parents=True, exist_ok=True)
    _csv(destination / 'rows.csv', all_rows)
    _csv(destination / 'per_volume.csv', volumes)
    _csv(destination / 'convergence.csv', all_stops)
    _csv(destination / 'paired_differences.csv', differences)
    _csv(destination / 'thresholds.csv', thresholds)
    if development is not None:
        _csv(destination / 'development_objectives.csv', development['rows'])
        atomic_json(destination / 'development_objective_summary.json', development)
    atomic_json(destination / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    result = summarize(args.output)
    print(json.dumps(dict(status=result['status'], coverage=result['coverage'],
        primary_joint_superiority=result['comparisons'][0]['confirmatory_joint_superiority']), sort_keys=True))


if __name__ == '__main__':
    main()
