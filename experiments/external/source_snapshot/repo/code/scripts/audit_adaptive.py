"""Independent physical and provenance audit of completed adaptive-study records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from adaptive_study import perturb
from adaptive_analysis import LAMBDA_GRID, FAILURE_THRESHOLDS_MM
from calibrate_adaptive import ROOT, OUT, SCENARIOS
from spine_registration.benchmark import atomic_json, fingerprint, global_to_local

FROZEN_CONFIG = ROOT / 'outputs/final_study/config.json'
MM_TOLERANCE = 1e-8
DEG_TOLERANCE = 1e-6


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def physical_metrics(pose, centres, landmarks, truth, *, global_pose=False):
    """Independent SciPy rotations; local translations move the native centroid."""
    centres, truth, pose = np.asarray(centres, float), np.asarray(truth, float), np.asarray(pose, float)
    if (centres.shape != (5, 3) or truth.shape != (5, 4, 4)
            or pose.shape != ((6,) if global_pose else (5, 6)) or len(landmarks) != 5
            or not all(np.isfinite(x).all() for x in (centres, truth, pose))):
        raise ValueError('Invalid physical pose, centres, or ground-truth dimensions')
    rotations = truth[:, :3, :3]
    if (not np.allclose(truth[:, 3, :], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-6)):
        raise ValueError('Ground-truth matrices are not proper rigid transforms')
    parameters = np.tile(pose, (5, 1)) if global_pose else pose
    estimates = np.tile(np.eye(4), (5, 1, 1))
    estimates[:, :3, :3] = Rotation.from_rotvec(parameters[:, :3]).as_matrix()
    estimates[:, :3, 3] = parameters[:, 3:]
    if not global_pose:
        estimates[:, :3, 3] += centres - np.einsum('nij,nj->ni', estimates[:, :3, :3], centres)
    vertebrae = []
    for index, (estimate, actual, centre, points) in enumerate(zip(estimates, truth, centres, landmarks)):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Native surface landmarks must be finite nonempty Nx3 arrays')
        predicted_points = points @ estimate[:3, :3].T + estimate[:3, 3]
        actual_points = points @ actual[:3, :3].T + actual[:3, 3]
        distances = np.linalg.norm(predicted_points - actual_points, axis=1)
        shift = (estimate @ np.r_[centre, 1] - actual @ np.r_[centre, 1])[:3]
        vertebrae.append(dict(label=20 + index, translation_error_mm=float(np.linalg.norm(shift)),
            rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(actual[:3, :3].T @ estimate[:3, :3]).magnitude())),
            surface_rms_tre_mm=float(np.sqrt(np.mean(distances ** 2))), surface_mtre_mm=float(distances.mean()),
            axis_error_mm=np.abs(shift).tolist()))
    metrics = {key: float(np.mean([x[key] for x in vertebrae])) for key in
               ('translation_error_mm', 'rotation_error_deg', 'surface_rms_tre_mm', 'surface_mtre_mm')}
    errors = np.array([x['translation_error_mm'] for x in vertebrae])
    metrics.update(case_failure_gt3mm=bool(np.any(errors > 3)),
                   vertebra_failure_rate_gt3mm=float(np.mean(errors > 3)), case_mean_gt3mm=bool(errors.mean() > 3))
    for threshold in FAILURE_THRESHOLDS_MM:
        metrics[f'failure_gt{threshold:g}mm'] = bool(np.any(errors > threshold))
    return dict(vertebrae=vertebrae, metrics=metrics)


def _close(actual, expected, tolerance, label, maxima):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f'Nonfinite or malformed {label}')
    delta = float(np.max(np.abs(actual - expected)))
    category = 'degrees' if label.endswith('_deg') else 'mm' if label.endswith('_mm') else 'dimensionless'
    maxima[category] = max(maxima.get(category, 0.), delta)
    if delta > tolerance:
        raise ValueError(f'{label} discrepancy {delta:.9g} exceeds {tolerance:g}')


def audit_record(row, case, metadata, *, maxima=None):
    """Check saved physical values and the objective identity, without rendering."""
    maxima = {} if maxima is None else maxima
    _close(metadata['centres'], case['centres'], MM_TOLERANCE, 'native_centres_mm', maxima)
    computed = physical_metrics(row['pose'], case['centres'], case['landmarks'], metadata['motion']['transforms'],
                                global_pose=row['method'] == 'A')
    if [x['label'] for x in row['vertebrae']] != list(range(20, 25)):
        raise ValueError('Saved records must contain five ordered vertebrae')
    for actual, expected in zip(row['vertebrae'], computed['vertebrae']):
        for key, value in expected.items():
            if key != 'label':
                _close(actual[key], value, DEG_TOLERANCE if key.endswith('_deg') else MM_TOLERANCE, key, maxima)
        for threshold in FAILURE_THRESHOLDS_MM:
            key = f'failure_gt{threshold:g}mm'
            if key in actual and actual[key] != bool(expected['translation_error_mm'] > threshold):
                raise ValueError(f'Vertebral threshold mismatch: {key}')
    for key, expected in computed['metrics'].items():
        # Only 3-mm failures are stored by the solver; all four are checked in report CSVs.
        if key.startswith('failure_gt') and key not in row['metrics']:
            continue
        if isinstance(expected, bool):
            if row['metrics'][key] not in (False, True) or bool(row['metrics'][key]) != expected:
                raise ValueError(f'Case failure definition mismatch: {key}')
        else:
            _close(row['metrics'][key], expected, DEG_TOLERANCE if key.endswith('_deg') else MM_TOLERANCE, key, maxima)
    weights, penalties = np.asarray(row['edge_weights'], float), np.asarray(row['edge_penalties'], float)
    if weights.shape != (4,) or penalties.shape != (4,) or not np.isfinite(weights).all() or not np.isfinite(penalties).all() or np.any(weights < 0) or np.any(penalties < 0):
        raise ValueError('Invalid edge weights/penalties')
    weighted = float(np.sum(weights * penalties) / 4)
    # The solver stores this diagnostic after a float32 mean of its edge penalties.
    _close(row['motion_penalty'], float(penalties.astype(np.float32).mean()), 1e-10, 'motion_penalty', maxima)
    _close(row['weighted_prior'], weighted, 1e-10, 'weighted_prior', maxima)
    if not np.isfinite(row['ncc_loss']):
        raise ValueError('Nonfinite saved image objective')
    _close(row['weighted_fraction'], weighted / max(row['ncc_loss'] + weighted, 1e-20), 1e-10, 'weighted_fraction', maxima)
    return computed


def _target(source, expected_config_hash, checked_sources, counts):
    source = Path(source)
    metadata = _json(source / 'metadata.json')
    if metadata['config_hash'] != expected_config_hash:
        raise ValueError('Target configuration hash mismatch')
    key = str(source.resolve())
    if key not in checked_sources and metadata['accepted']:
        for view in ('AP', 'LAT'):
            image = np.load(source / f'{view}.npy', allow_pickle=False)
            if image.shape != (256, 256) or not np.isfinite(image).all():
                raise ValueError('Target must be a finite 256x256 array')
            if hashlib.sha256(image.tobytes()).hexdigest() != metadata['image_hashes'][view]:
                raise ValueError('Target array checksum mismatch')
            counts['target_arrays_checked'] += 1
        checked_sources.add(key)
    return metadata


def _configuration(root):
    config, split = _json(root / 'config.json'), _json(root / 'split.json')
    base = _json(FROZEN_CONFIG)
    config_hash = fingerprint(config)
    if split != config['split'] or fingerprint(base) != config['base_config_hash']:
        raise ValueError('Configuration, frozen baseline, or split mismatch')
    if (len(split['development']) != 25 or len(split['evaluation']) != 50
            or len(set(split['development'] + split['evaluation'])) != 75
            or set(base['cases']) != set(split['development'] + split['evaluation'])):
        raise ValueError('Expected exact disjoint 25/50 partition of the 75 baseline volumes')
    if 'input_manifest_hash' in base and fingerprint(_json(ROOT / 'inputs/ppcnet75/manifest.json')) != base['input_manifest_hash']:
        raise ValueError('Original input manifest changed')
    if tuple(config['lambda_grid']) != LAMBDA_GRID or tuple(config['main_scenarios']) != SCENARIOS:
        raise ValueError('Frozen analysis grid/scenarios changed')
    for sources in (base['source_hashes'], config['source_hashes']):
        for relative, digest in sources.items():
            if _sha(ROOT / relative) != digest:
                raise ValueError(f'Implementation changed: {relative}')
    selection = _json(root / 'calibration/selection.json')
    fixed, gate = _json(root / 'calibration/fixed_selection.json'), _json(root / 'calibration/proxy_gate.json')
    if (any(x.get('config_hash') != config_hash for x in (selection, fixed, gate))
            or selection.get('fixed_selection_hash') != fingerprint(fixed)
            or selection.get('proxy_gate_hash') != fingerprint(gate)
            or selection.get('fixed_lambda') != fixed.get('fixed_lambda')
            or selection['fixed_lambda'] not in LAMBDA_GRID
            or bool(selection.get('adaptive') is not None) != bool(gate['gate_passed'])):
        raise ValueError('Final calibration selection integrity mismatch')
    return config, config_hash, split, base, selection


def _signature(row, records, bank, native_case, config, base):
    method = row['method']
    imported_digest = None
    if 'imported_provenance' in row:
        provenance = row['imported_provenance']
        source = (ROOT / provenance['path']).resolve()
        if not source.is_relative_to(ROOT.resolve()) or _sha(source) != provenance['sha256']:
            raise ValueError('Imported result file digest changed')
        original = _json(source)
        expected_method = 'B' if method.startswith('start_') else method
        if (original.get('case') != row['case'] or original.get('method') != expected_method
                or original.get('target_hash') != row['target_hash']
                or original.get('config_hash') != config['base_config_hash']
                or not np.array_equal(original['pose'], row['pose'])):
            raise ValueError('Imported result identity/target/pose differs')
        imported_digest = provenance['sha256']
    if method in ('A', 'Zero'):
        initial = None
    elif method.startswith('start_'):
        k = int(method[6:])
        if k not in range(5):
            raise ValueError('Candidate start index outside the locked five-start bank')
        initial = global_to_local(np.asarray(records['A']['pose']), native_case['centres'])
        if k:
            index = base['cases'].index(row['case'])
            initial = perturb(initial, config['seed'] + index * 1009 + k * 7919)
    else:
        initial = np.asarray(records[f"start_{bank['best']}"]['pose'])
    signature = fingerprint(dict(method=method, case=row['case'], weights=row['edge_weights'], scales=row['scales'],
        initial=None if initial is None else np.asarray(initial).tolist(), imported_sha256=imported_digest,
        implementation_sha256=config['source_hashes']['code/scripts/adaptive_study.py']))
    if signature != row['input_signature']:
        raise ValueError('Checkpoint input signature mismatch')


def _oof_targets(root, config, config_hash, split, selection):
    """Reconstruct the locked OOF plan from eligible training-bank proxies only."""
    if selection.get('adaptive') is None:
        return {}
    manifest = _json(root / 'calibration/adaptive_oof_manifest.json')
    fixed = _json(root / 'calibration/fixed_selection.json')
    gate = _json(root / 'calibration/proxy_gate.json')
    provenance = selection['oof_provenance']
    expected = dict(version='adaptive_oof_v1', config_hash=config_hash,
        fixed_lambda=selection['fixed_lambda'], fixed_selection_hash=fingerprint(fixed),
        proxy_gate_hash=fingerprint(gate), frozen_source_hashes=config['source_hashes'],
        lambda_grid=list(LAMBDA_GRID), scenarios=list(SCENARIOS), optimizer=config['optimizer'],
        levels=config['levels'], zero_lambda='reuse strong B')
    if (any(manifest.get(key) != value for key, value in expected.items())
            or gate.get('gate_passed') is not True or fixed.get('status') != 'FIXED_CALIBRATION_COMPLETE'
            or provenance['manifest_hash'] != fingerprint(manifest)
            or Path(provenance['manifest_path']).resolve() != (root / 'calibration/adaptive_oof_manifest.json').resolve()
            or provenance['source_hashes'] != manifest['source_hashes']
            or set(manifest['source_hashes']) != {'code/scripts/adaptive_oof.py'}
            or provenance['zero_lambda'] != manifest['zero_lambda']):
        raise ValueError('OOF manifest/calibration provenance mismatch')
    for relative, digest in manifest['source_hashes'].items():
        if _sha(ROOT / relative) != digest:
            raise ValueError(f'OOF implementation changed: {relative}')
    included, excluded, banks, eligibility = [], [], {}, {}
    expected_inputs = {str(root / 'config.json'), str(root / 'split.json')}
    for case in split['development']:
        statuses = []
        for scenario in SCENARIOS:
            path = root / 'eligibility' / scenario / f'{case}.json'
            expected_inputs.add(str(path))
            eligible = _json(path)
            if eligible.get('case') != case or type(eligible.get('accepted')) is not bool:
                raise ValueError('OOF development eligibility identity/status mismatch')
            statuses.append(eligible['accepted'])
            eligibility[case, scenario] = eligible
            if not eligible['accepted']:
                continue
            expected_inputs.add(str(Path(eligible['source']) / 'metadata.json'))
            for method in ['A', 'Zero', 'B'] + [f'lambda_{w:g}' for w in LAMBDA_GRID[1:]]:
                expected_inputs.add(str(root / 'results' / scenario / case / f'{method}.json'))
            for method in ['selection'] + [f'start_{k}' for k in range(5)]:
                expected_inputs.add(str(root / 'banks' / scenario / case / f'{method}.json'))
            bank = _json(root / 'banks' / scenario / case / 'selection.json')
            edges = np.asarray(bank['proxy_edges'], float)
            if edges.shape != (4,) or not np.isfinite(edges).all() or np.any(edges < 0):
                raise ValueError('Invalid OOF training-bank proxy')
            banks[case, scenario] = bank
        if len(set(statuses)) != 1:
            raise ValueError('OOF development eligibility differs across scenarios')
        (included if statuses[0] else excluded).append(case)
    for artifact in (manifest, fixed, gate, selection):
        if artifact.get('included_cases') != included or artifact.get('excluded_cases') != excluded:
            raise ValueError('OOF development cohort mismatch')
    inputs = manifest['development_provenance']
    if inputs != fixed['provenance'] or set(inputs) != expected_inputs:
        raise ValueError('OOF development provenance coverage mismatch')
    for path, digest in inputs.items():
        if _sha(path) != digest:
            raise ValueError(f'OOF development input changed: {path}')
    folds = split['development_folds']
    if len(folds) != 5 or sorted(sum(folds, [])) != sorted(split['development']):
        raise ValueError('OOF requires five complete development folds')
    planned, fold_for = [], {}
    for index, validation in enumerate(folds):
        training = [case for case in included if case not in validation]
        held = [case for case in included if case in validation]
        if len(training) < 2 or not held:
            raise ValueError('OOF fold lacks eligible training/validation volumes')
        values = np.asarray([banks[case, scenario]['proxy_edges'] for case in training for scenario in SCENARIOS])
        fold = dict(fold=index, training_cases=training, validation_cases=held,
            tau=max(float(np.median(values)), 1e-12), training_edge_observations=int(values.size))
        planned.append(fold)
        fold_for.update({case: fold for case in held})
    if manifest['folds'] != planned or provenance['folds'] != planned:
        raise ValueError('OOF folds/tau differ from training-only bank proxies')
    targets, result_paths = {}, set()
    for case in included:
        fold = fold_for[case]
        for scenario in SCENARIOS:
            bank = banks[case, scenario]
            edges = np.asarray(bank['proxy_edges'], float)
            winner = root / 'banks' / scenario / case / f"start_{bank['best']}.json"
            weights = {f'adaptive_oof_{w:g}': (w * edges / (edges + fold['tau'])).tolist() for w in LAMBDA_GRID[1:]}
            targets[case, scenario] = dict(case=case, scenario=scenario, fold=fold['fold'], tau=fold['tau'],
                target_hash=eligibility[case, scenario]['target_hash'], bank_winner=str(winner),
                bank_winner_sha256=inputs[str(winner)], proxy_edges=edges.tolist(), edge_weights=weights)
            result_paths.update(str(root / 'results' / scenario / case / f'{method}.json') for method in weights)
    all_edges = [banks[case, scenario]['proxy_edges'] for case in included for scenario in SCENARIOS]
    if (manifest['targets'] != list(targets.values())
            or set(provenance['result_hashes']) != result_paths
            or provenance['final_tau_edge_observations'] != len(included) * len(SCENARIOS) * 4
            or selection['adaptive']['tau'] != max(float(np.median(all_edges)), 1e-12)
            or selection['adaptive']['lambda_max'] not in LAMBDA_GRID
            or selection['adaptive']['lambda_max'] != selection['selection']['selected_lambda']):
        raise ValueError('OOF target/result coverage, weights, or final rule mismatch')
    return targets


def _report_checks(root, computed, maxima, counts, split=None, selection=None):
    path = root / 'summary/rows.csv'
    if not path.is_file():
        return 'not_present'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        methods = {'A': 'A', 'Zero': 'Zero', 'B': 'B', 'C_lambda0.1': 'lambda_0.1',
                   'C_fixed': 'B' if selection['fixed_lambda'] == 0 else f"lambda_{selection['fixed_lambda']:g}"}
        if selection.get('adaptive') is not None:
            methods['C_adaptive'] = 'C_adaptive'
        expected = {(case, scenario, method, source) for case in split['evaluation']
                    if (case, 'clean', 'B') in computed for scenario in SCENARIOS for method, source in methods.items()}
        actual = [(x['case'], x['scenario'], x['method'], x['source_method']) for x in rows if x['dataset'] == 'main']
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Main report record coverage/aliases differ from the audited evaluation cohort')
    for row in rows:
        key = (row['case'], row['scenario'], row['source_method'])
        if key not in computed:
            raise ValueError('Report references an unaudited physical record')
        actual = computed[key]['metrics']
        for metric in ('surface_rms_tre_mm', 'translation_error_mm', 'rotation_error_deg'):
            _close(float(row[metric]), actual[metric], DEG_TOLERANCE if metric.endswith('_deg') else MM_TOLERANCE,
                   metric, maxima)
        for threshold in FAILURE_THRESHOLDS_MM:
            field = f'failure_gt{threshold:g}mm'
            if int(row[field]) != int(actual[field]):
                raise ValueError(f'Report threshold mismatch: {field}')
            counts['report_threshold_flags_checked'] += 1
    path = root / 'summary/thresholds.csv'
    if not path.is_file():
        raise ValueError('Threshold summary missing beside rows.csv')
    with path.open(newline='') as handle:
        summaries = list(csv.DictReader(handle))
    for summary in summaries:
        dataset = summary['dataset']
        selected = [x for x in rows if x['method'] == summary['method'] and
                    ((x['dataset'] == 'main' and ((x['scenario'] == 'clean') == (dataset == 'clean')))
                     if dataset in ('clean', 'combined') else x['dataset'] == dataset)]
        if not selected:
            raise ValueError('Threshold summary has no matching audited report rows')
        field = f"failure_gt{float(summary['threshold_mm']):g}mm"
        ids = sorted({x['case'] for x in selected})
        frequencies = [np.mean([computed[(x['case'], x['scenario'], x['source_method'])]['metrics'][field]
                               for x in selected if x['case'] == case]) for case in ids]
        if len(ids) != int(summary['n_eligible']):
            raise ValueError('Threshold summary independent-volume denominator mismatch')
        _close(float(summary['mean_failure_frequency']), float(np.mean(frequencies)), 1e-12,
               'failure_frequency', maxima)
        _close(float(summary['pipeline_success_frequency_all_assigned']),
               float(np.sum(1 - np.asarray(frequencies)) / int(summary['n_assigned'])), 1e-12,
               'pipeline_success_frequency', maxima)
        counts['report_threshold_summaries_checked'] += 1
    return 'checked'


def audit_outputs(output_root=OUT, *, case_loader=None):
    """Audit all saved methods and require complete main-study record coverage."""
    root = Path(output_root)
    if case_loader is None:
        from run_final_study import verified_case
        case_loader = verified_case
    counts = dict(records_found=0, records_checked=0, result_records_checked=0, bank_records_checked=0, oof_records_checked=0,
        native_cases_loaded=0, target_arrays_checked=0, rejected_targets_checked=0, missing_records=0,
        report_threshold_flags_checked=0, report_threshold_summaries_checked=0)
    maxima, errors, checked_sources, computed = {}, [], set(), {}
    failure_count, report_status = 0, 'not_checked'
    record_digest = hashlib.sha256()

    def fail(message):
        nonlocal failure_count
        failure_count += 1
        if len(errors) < 20:
            errors.append(str(message))

    try:
        config, config_hash, split, base, selection = _configuration(root)
        oof_targets = _oof_targets(root, config, config_hash, split, selection)
        assigned = split['development'] + split['evaluation']
        manifest = _json(root / 'robustness/manifest.json') if (root / 'robustness/manifest.json').is_file() else None
        if manifest is not None:
            if (manifest['config_hash'] != config_hash or manifest['cases'] != split['evaluation'][:10]
                    or manifest['selection_sha256'] != _sha(root / 'calibration/selection.json')
                    or manifest['split_sha256'] != _sha(root / 'split.json')):
                raise ValueError('Supplementary manifest provenance mismatch')
            for relative, digest in manifest['source_hashes'].items():
                if _sha(ROOT / relative) != digest:
                    raise ValueError('Supplementary source changed')
        paths = sorted(path for section in ('results', 'banks') for path in (root / section).glob('*/*/*.json')
                       if not (section == 'banks' and path.name == 'selection.json'))
        counts['records_found'] = len(paths)
        by_case = {name: {} for name in assigned}
        for path in paths:
            case, scenario = path.parent.name, path.parent.parent.name
            record_digest.update(f'{path.relative_to(root)}:{_sha(path)}\n'.encode())
            if case not in by_case:
                fail(f'Unassigned record volume: {path}'); continue
            by_case[case].setdefault(scenario, []).append(path)
        for name in assigned:
            native = None
            scenarios = set(SCENARIOS) | set(by_case[name])
            if manifest is not None and name in manifest['cases']:
                scenarios |= (set(manifest['motion_scenarios']) | set(manifest['corruption_scenarios'])
                              | set(manifest['amplitude_scenarios']))
            if any(root.glob('sensitivity_*_completion.json')) and name in split['evaluation'][:10]:
                scenarios |= {'siddon_clean', 'siddon_combined'}
            main_statuses, main_rejections = [], []
            for scenario in sorted(scenarios):
                try:
                    eligibility = _json(root / 'eligibility' / scenario / f'{name}.json')
                    if eligibility.get('case') != name or type(eligibility.get('accepted')) is not bool:
                        raise ValueError('Invalid target eligibility identity/status')
                    if scenario in SCENARIOS:
                        main_statuses.append(eligibility['accepted'])
                    source = Path(eligibility['source'])
                    if not (source / 'metadata.json').exists() and not eligibility['accepted']:
                        source = Path(_json(root / 'eligibility/clean' / f'{name}.json')['source'])
                    metadata = _target(source, config['base_config_hash'], checked_sources, counts)
                    if source.name != name or metadata.get('case') != name or metadata.get('accepted') != eligibility['accepted']:
                        raise ValueError('Target metadata identity/eligibility mismatch')
                    if manifest is not None and scenario in (set(manifest['motion_scenarios'])
                            | set(manifest['corruption_scenarios']) | set(manifest['amplitude_scenarios'])):
                        if metadata.get('robustness_manifest_hash') != fingerprint(manifest):
                            raise ValueError('Supplementary target manifest hash mismatch')
                    if not eligibility['accepted']:
                        if (metadata['motion'].get('accepted') is not False
                                or fingerprint(metadata['motion']) != fingerprint(eligibility['motion'])):
                            raise ValueError('Geometry rejection evidence mismatch')
                        if scenario in SCENARIOS:
                            main_rejections.append(fingerprint(metadata['motion']))
                        if by_case[name].get(scenario):
                            raise ValueError('Results exist for a rejected target')
                        counts['rejected_targets_checked'] += 1
                        continue
                    expected_scenario = {'clean': 'moderate', 'combined_s0': 'combined'}.get(scenario, scenario)
                    if metadata.get('scenario') != expected_scenario or fingerprint(metadata) != eligibility.get('target_hash'):
                        raise ValueError('Target metadata scenario/fingerprint mismatch')
                    weights = LAMBDA_GRID if name in split['development'] else sorted({selection['fixed_lambda'], .1})
                    expected = {'A', 'Zero', 'B'} | {f'lambda_{w:g}' for w in weights if w}
                    if name in split['evaluation'] and selection.get('adaptive') is not None:
                        expected.add('C_adaptive')
                    if (name, scenario) in oof_targets:
                        expected.update(oof_targets[name, scenario]['edge_weights'])
                    if manifest is not None and name in manifest['cases'] and scenario in manifest['prior_scale_scenarios']:
                        expected |= {f'C_scale_t{x[1]:g}' for x in manifest['prior_scales']}
                    expected_paths = [root / 'results' / scenario / name / f'{method}.json' for method in expected]
                    expected_paths += [root / 'banks' / scenario / name / f'start_{k}.json' for k in range(5)]
                    expected_paths += [root / 'banks' / scenario / name / 'selection.json']
                    missing = [str(path) for path in expected_paths if not path.is_file()]
                    if missing:
                        counts['missing_records'] += len(missing)
                        raise ValueError(f'Incomplete target: {len(missing)} missing records; first {missing[0]}')
                    if native is None:
                        native = case_loader(name)
                        counts['native_cases_loaded'] += 1
                    records, record_paths = {}, {}
                    for path in by_case[name][scenario]:
                        row = _json(path)
                        # Optimization traces are unnecessary for the independent metric audit.
                        if not path.stem.startswith('adaptive_oof_'):
                            row.pop('optimization', None)
                        method = path.stem
                        if method in records:
                            raise ValueError('Duplicate method record across results/banks')
                        records[method], record_paths[method] = row, path
                    bank = _json(root / 'banks' / scenario / name / 'selection.json')
                    losses = [records[f'start_{k}']['ncc_loss'] for k in range(5)]
                    if (any(bank.get(k) != v for k, v in dict(case=name, scenario=scenario, config_hash=config_hash,
                            target_hash=fingerprint(metadata)).items()) or not np.isfinite(losses).all()
                            or bank['best'] != int(np.argmin(losses)) or not np.array_equal(bank['ncc_losses'], losses)):
                        raise ValueError('Candidate selection integrity mismatch')
                    proxy = np.asarray(bank['proxy_edges'], float)
                    if proxy.shape != (4,) or not np.isfinite(proxy).all() or np.any(proxy < 0) or bank['proxy_case'] != float(proxy.max()):
                        raise ValueError('Invalid saved candidate dispersion proxy')
                    for method, row in records.items():
                        try:
                            if any(row.get(k) != v for k, v in dict(case=name, scenario=scenario, method=method,
                                    config_hash=config_hash, target_hash=fingerprint(metadata),
                                    implementation_sha256=config['source_hashes']['code/scripts/adaptive_study.py']).items()):
                                raise ValueError('Record identity/configuration/target/source mismatch')
                            expected_weights = np.zeros(4)
                            expected_scales = [3., 3.]
                            if method.startswith('lambda_'):
                                expected_weights[:] = float(method[7:])
                            elif method.startswith('C_scale_t'):
                                expected_weights[:] = selection['fixed_lambda']
                                expected_scales[1] = float(method[9:])
                            elif method == 'C_adaptive':
                                rule = selection['adaptive']
                                expected_weights = rule['lambda_max'] * proxy / (proxy + rule['tau'])
                            elif method.startswith('adaptive_oof_'):
                                target = oof_targets[name, scenario]
                                expected_weights = target['edge_weights'][method]
                                if (_sha(record_paths[method]) != selection['oof_provenance']['result_hashes'][str(record_paths[method])]
                                        or [level['level'] for level in row['optimization']['levels']] != config['levels']):
                                    raise ValueError('OOF saved result digest/optimization levels mismatch')
                            elif method not in ('A', 'Zero', 'B') and not method.startswith('start_'):
                                raise ValueError('Unrecognized method for frozen-weight validation')
                            if row['scales'] != expected_scales or not np.allclose(row['edge_weights'], expected_weights, rtol=1e-10, atol=1e-12):
                                raise ValueError('Saved weights/scales differ from frozen method')
                            _signature(row, records, bank, native, config, base)
                            computed[(name, scenario, method)] = audit_record(row, native, metadata, maxima=maxima)
                            counts['records_checked'] += 1
                            counts['oof_records_checked'] += int(method.startswith('adaptive_oof_'))
                            section = 'bank' if record_paths[method].parent.parent.parent.name == 'banks' else 'result'
                            counts[f'{section}_records_checked'] += 1
                        except Exception as exc:
                            fail(f'{record_paths[method].relative_to(root)}: {type(exc).__name__}: {exc}')
                except Exception as exc:
                    fail(f'{name}/{scenario}: {type(exc).__name__}: {exc}')
            if len(main_statuses) != 6 or len(set(main_statuses)) != 1 or len(set(main_rejections)) > 1:
                fail(f'{name}: main eligibility/rejection evidence is not common to all six scenarios')
        if counts['records_checked'] != counts['records_found']:
            fail('Not every discovered physical record passed audit')
        if not counts['records_checked']:
            fail('No physical records were audited')
        try:
            report_status = _report_checks(root, computed, maxima, counts, split, selection)
        except Exception as exc:
            fail(f'Report tables: {type(exc).__name__}: {exc}')
    except Exception as exc:
        fail(f'Configuration/setup: {type(exc).__name__}: {exc}')
    result = dict(status='PASS' if failure_count == 0 else 'FAIL', counts=counts,
        maximum_absolute_discrepancies=maxima, tolerances=dict(mm=MM_TOLERANCE, degrees=DEG_TOLERANCE),
        failure_count=failure_count, first_errors=errors, errors_truncated=failure_count > len(errors),
        record_manifest_sha256=record_digest.hexdigest(), audit_source_sha256=_sha(__file__),
        report_threshold_audit=report_status,
        independence='NumPy/SciPy physical transforms and native surface points; no benchmark.evaluate or extra_metrics.',
        scope='All saved result/bank methods; complete main targets and declared robustness targets required.',
        not_reaudited='Forward-rendered NCC numerics, optimizer convergence, and individual edge-penalty numerics were not independently recomputed; weighted prior/fraction identities were checked.')
    destination = root / 'audit'
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'independent_audit.json', result)
    lines = [f'# Independent physical/data audit: {result["status"]}', '',
        f'Records passed: {counts["records_checked"]}/{counts["records_found"]}; native cases loaded: {counts["native_cases_loaded"]}; target arrays checked: {counts["target_arrays_checked"]}.',
        f'Maximum absolute discrepancies: {json.dumps(maxima, sort_keys=True)}.',
        f'Tolerances: {MM_TOLERANCE:g} mm and {DEG_TOLERANCE:g} degrees. Failures: {failure_count}.', '', result['not_reaudited']]
    if errors:
        lines += ['', 'First failures:'] + [f'- {error}' for error in errors]
    (destination / 'independent_audit.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    result = audit_outputs(args.output)
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'maximum_absolute_discrepancies', 'failure_count')}, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
