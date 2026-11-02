"""Local deterministic replay of B/fixed/adaptive C on one case, two targets."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import adaptive_study as study
from audit_adaptive import physical_metrics
from calibrate_adaptive import _record
from summarize_adaptive import _identity
from spine_registration.benchmark import ImageProblem, atomic_json, fingerprint

TOLERANCES = dict(physical_mm=1e-4, rotation_deg=1e-4, loss=1e-6)
SCENARIOS = ('clean', 'combined_s0')


def _matrices(pose, centres):
    pose, centres = np.asarray(pose, float), np.asarray(centres, float)
    if pose.shape != (5, 6) or centres.shape != (5, 3) or not np.isfinite(pose).all():
        raise ValueError('Replay requires five finite centroid-local poses')
    matrices = np.tile(np.eye(4), (5, 1, 1))
    matrices[:, :3, :3] = Rotation.from_rotvec(pose[:, :3]).as_matrix()
    matrices[:, :3, 3] = centres + pose[:, 3:] - np.einsum('nij,nj->ni', matrices[:, :3, :3], centres)
    return matrices


def compare_replay(saved, observed_pose, observed_ncc, native_case, truth):
    """Compare physical points, pose rotation, reported GT errors, and final NCC."""
    centres, points = native_case['centres'], native_case['landmarks']
    old, new = _matrices(saved['pose'], centres), _matrices(observed_pose, centres)
    changed = physical_metrics(observed_pose, centres, points, old)
    observed = physical_metrics(observed_pose, centres, points, truth)
    maximum_point_change = max(float(np.linalg.norm(
        np.asarray(p) @ (a[:3, :3] - b[:3, :3]).T + a[:3, 3] - b[:3, 3], axis=1).max())
        for p, a, b in zip(points, new, old))
    expected_errors = {key: [v[key] for v in saved['vertebrae']] for key in
                       ('surface_rms_tre_mm', 'translation_error_mm', 'rotation_error_deg')}
    observed_errors = {key: [v[key] for v in observed['vertebrae']] for key in expected_errors}
    differences = dict(max_surface_point_change_mm=maximum_point_change,
        max_centroid_change_mm=max(v['translation_error_mm'] for v in changed['vertebrae']),
        max_translation_parameter_change_mm=float(np.linalg.norm(np.asarray(observed_pose)[:, 3:] - np.asarray(saved['pose'])[:, 3:], axis=1).max()),
        max_pose_rotation_change_deg=max(v['rotation_error_deg'] for v in changed['vertebrae']),
        ncc_loss=abs(float(observed_ncc) - saved['ncc_loss']))
    for key in expected_errors:
        differences[f'max_{key}'] = float(np.max(np.abs(np.asarray(expected_errors[key]) - observed_errors[key])))
    passed = all(np.isfinite(value) and value <= (TOLERANCES['rotation_deg'] if key.endswith('_deg') else
                 TOLERANCES['loss'] if key == 'ncc_loss' else TOLERANCES['physical_mm']) for key, value in differences.items())
    return dict(status='PASS' if passed else 'FAIL', expected=dict(ncc_loss=saved['ncc_loss'], **expected_errors),
                observed=dict(ncc_loss=float(observed_ncc), **observed_errors), maximum_differences=differences)


def check_objective(saved, observed):
    """The same-renderer numerical check is not an independent mathematical audit."""
    expected = {key: saved[key] for key in ('ncc_loss', 'edge_penalties', 'motion_penalty', 'weighted_prior', 'weighted_fraction')}
    differences = {}
    for key in expected:
        a, b = np.asarray(expected[key], float), np.asarray(observed[key], float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Malformed or nonfinite objective check: {key}')
        differences[key] = float(np.max(np.abs(a - b)))
    return dict(status='PASS' if max(differences.values()) <= TOLERANCES['loss'] else 'FAIL',
                expected=expected, observed=observed, maximum_differences=differences)


def refinement_jobs(fixed_lambda, adaptive=None, proxy_edges=None):
    jobs = [('B', 0., ['B', 'C_fixed'])] if fixed_lambda == 0 else [
        ('B', 0., ['B']), (f'lambda_{fixed_lambda:g}', fixed_lambda, ['C_fixed'])]
    if adaptive is not None:
        edges = np.asarray(proxy_edges, float)
        if edges.shape != (4,) or not np.isfinite(edges).all() or np.any(edges < 0):
            raise ValueError('Invalid frozen adaptive proxy edges')
        weights = adaptive['lambda_max'] * edges / (edges + adaptive['tau'])
        jobs.append(('C_adaptive', weights.tolist(), ['C_adaptive']))
    return jobs


def _objective(problem, pose, weights, scales):
    with torch.no_grad():
        tensor = torch.as_tensor(pose, dtype=torch.float32, device=problem.centres.device)
        penalties = study.edge_penalties(tensor, problem.centres, *scales).cpu().numpy()
        ncc = float(problem.image_loss(tensor, 256))
    weighted = float(np.mean(np.asarray(weights, float) * penalties))
    return dict(ncc_loss=ncc, edge_penalties=penalties.tolist(), motion_penalty=float(penalties.mean()),
                weighted_prior=weighted, weighted_fraction=weighted / max(ncc + weighted, 1e-20))


def _first_eligible(root, ordered_ids):
    for case in ordered_ids:
        statuses = [json.loads((root / 'eligibility' / scenario / f'{case}.json').read_text()) for scenario in SCENARIOS]
        if any(row.get('case') != case or type(row.get('accepted')) is not bool for row in statuses):
            raise ValueError('Malformed replay eligibility')
        if statuses[0]['accepted'] != statuses[1]['accepted']:
            raise ValueError('Clean/combined eligibility differs')
        if statuses[0]['accepted']:
            return case, dict(zip(SCENARIOS, statuses))
    raise ValueError('No eligible evaluation volume for deterministic replay')


def replay():
    root, trials, errors, case = study.OUT, [], [], None
    started = time.perf_counter()
    try:
        # Must be first: validate the frozen solver configuration before any replay.
        config, config_hash, base, base_hash = study.initialize()
        torch.set_num_threads(1)
        checked, checked_hash, split, selection, _, _ = _identity(root, {})
        if checked != config or checked_hash != config_hash:
            raise ValueError('Configuration changed during replay setup')
        summary = json.loads((root / 'summary/summary.json').read_text())
        if summary.get('status') != 'COMPLETE' or summary.get('config_hash') != config_hash or summary.get('selection') != selection:
            raise ValueError('Require matching COMPLETE final evaluation summary before replay')
        case, eligibility = _first_eligible(root, split['evaluation'])
        native = study.verified_case(case)
        problem = ImageProblem(native, base)
        runner_hash = config['source_hashes']['code/scripts/adaptive_study.py']
        for scenario in SCENARIOS:
            try:
                source = Path(eligibility[scenario]['source'])
                targets, metadata = study.checked_target(source, base_hash)
                expected_scenario = 'moderate' if scenario == 'clean' else 'combined'
                if (targets is None or source.name != case or metadata.get('case') != case
                        or metadata.get('scenario') != expected_scenario or fingerprint(metadata) != eligibility[scenario]['target_hash']
                        or any(tuple(image.shape) != (256, 256) for image in targets.values())):
                    raise ValueError('Replay target identity/shape/hash mismatch')
                problem.set_targets(targets)
                bank_directory = root / 'banks' / scenario / case
                bank = json.loads((bank_directory / 'selection.json').read_text())
                candidates = [_record(bank_directory / f'start_{k}.json', f'start_{k}', case, scenario,
                    fingerprint(metadata), config_hash, runner_hash, {}) for k in range(5)]
                losses = [row['ncc_loss'] for row in candidates]
                if (any(bank.get(key) != value for key, value in dict(case=case, scenario=scenario,
                        target_hash=fingerprint(metadata), config_hash=config_hash).items())
                        or bank['best'] != int(np.argmin(losses)) or not np.array_equal(bank['ncc_losses'], losses)):
                    raise ValueError('Shared candidate-bank winner is inconsistent')
                initial = np.asarray(candidates[bank['best']]['pose'])
                for method, weight, aliases in refinement_jobs(selection['fixed_lambda'], selection.get('adaptive'), bank.get('proxy_edges')):
                    try:
                        path = root / 'results' / scenario / case / f'{method}.json'
                        saved = _record(path, method, case, scenario, fingerprint(metadata), config_hash, runner_hash, {})
                        weights = np.broadcast_to(weight, (4,)).tolist()
                        signature = fingerprint(dict(method=method, case=case, weights=weights,
                            scales=[3., 3.], initial=initial.tolist(), imported_sha256=None, implementation_sha256=runner_hash))
                        if saved['input_signature'] != signature or saved['edge_weights'] != weights or saved['scales'] != [3., 3.]:
                            raise ValueError('Saved refinement inputs differ from frozen shared start/weights')
                        torch.cuda.synchronize()
                        trial_start = time.perf_counter()
                        observed_pose, stop = study.refine(problem, initial, saved['edge_weights'], base, saved['scales'])
                        torch.cuda.synchronize()
                        refinement_seconds = time.perf_counter() - trial_start
                        observed_objective = _objective(problem, observed_pose, saved['edge_weights'], saved['scales'])
                        comparison = compare_replay(saved, observed_pose, observed_objective['ncc_loss'], native,
                                                    np.asarray(metadata['motion']['transforms']))
                        # Reevaluate the saved final pose as well as the replayed final pose.
                        check_start = time.perf_counter()
                        saved_objective = _objective(problem, np.asarray(saved['pose']), saved['edge_weights'], saved['scales'])
                        torch.cuda.synchronize()
                        objective_check = check_objective(saved, saved_objective)
                        trials.append(dict(case=case, scenario=scenario, source_method=method, aliases=aliases,
                            target_hash=fingerprint(metadata), saved_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bank_winner=bank['best'], edge_weights=saved['edge_weights'], prior_scales=saved['scales'],
                            refinement_runtime_seconds=refinement_seconds,
                            saved_objective_recheck_seconds=time.perf_counter() - check_start,
                            stopping={key: stop[key] for key in ('stop_reason', 'converged', 'hit_max_iterations', 'iterations', 'evaluations')},
                            stopping_by_level=[{key: level[key] for key in ('level', 'stop_reason', 'converged', 'hit_max_iterations', 'iterations', 'evaluations')}
                                               for level in stop['levels']],
                            comparison=comparison, saved_final_objective_check=objective_check,
                            replayed_final_objective=observed_objective))
                        if comparison['status'] != 'PASS' or objective_check['status'] != 'PASS':
                            errors.append(f'{case}/{scenario}/{method}: predefined replay tolerance exceeded')
                    except Exception as exc:
                        errors.append(f'{case}/{scenario}/{method}: {type(exc).__name__}: {exc}')
            except Exception as exc:
                errors.append(f'{case}/{scenario}: {type(exc).__name__}: {exc}')
        expected_trials = len(SCENARIOS) * (len(refinement_jobs(selection['fixed_lambda'])) + int(selection.get('adaptive') is not None))
        if len(trials) != expected_trials:
            errors.append(f'Incomplete replay: {len(trials)}/{expected_trials} refinements checked')
    except Exception as exc:
        errors.append(f'Setup: {type(exc).__name__}: {exc}')
    result = dict(status='PASS' if not errors else 'FAIL', case=case, scenarios=list(SCENARIOS),
        tolerances=TOLERANCES, trials=trials, errors=errors, total_runtime_seconds=time.perf_counter() - started,
        replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='One volume, two targets: B/fixed C and, when enabled, adaptive C local deterministic reproducibility check, not all-cohort replay. Frozen renderer/prior reused; objective recheck is not independent mathematics.',
        unchanged='Stored targets, banks and result records are read only. A is not rerun; its previous corrected replay audit remains separate.')
    atomic_json(root / 'audit/replay.json', study.plain(result))
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = replay()
    print(json.dumps(dict(status=result['status'], case=result['case'], refinements=len(result['trials']),
                         runtime_seconds=result['total_runtime_seconds'], errors=result['errors']), sort_keys=True))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
