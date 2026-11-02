"""Checkpointed weight calibration; frozen original study is read only."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import portable_lock as fcntl
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from spine_registration.benchmark import ImageProblem, atomic_json, evaluate, fingerprint, global_to_local
from spine_registration.benchmark_optimizers import optimize_torch
from spine_registration.torch_registration import local_motion_tensor, se3_log
from run_final_study import frozen_config, verified_case, optimize_A, save_targets
from final_study_support import degrade, extra_metrics, sha256
from generate_posture_dataset import plain
OUT = ROOT / 'outputs/adaptive_prior_study'
OLD = ROOT / 'outputs/final_study'
GRID = (0., .003, .01, .03, .1)


def relative_frames(params, centres):
    offsets = torch.eye(4, dtype=params.dtype, device=params.device).repeat(5, 1, 1)
    offsets[:, :3, 3] = centres
    frames = local_motion_tensor(params, centres) @ offsets
    return torch.linalg.solve(frames[:-1], frames[1:])


def edge_penalties(params, centres, rotation_scale_deg=3., translation_scale_mm=3.):
    if min(rotation_scale_deg, translation_scale_mm) <= 0:
        raise ValueError('Positive prior scales required')
    offsets = torch.eye(4, dtype=params.dtype, device=params.device).repeat(5, 1, 1)
    offsets[:, :3, 3] = centres
    pre = torch.linalg.solve(offsets[:-1], offsets[1:])
    twist = se3_log(torch.linalg.solve(pre, relative_frames(params, centres)))
    return (twist[:, :3] / np.deg2rad(rotation_scale_deg)).square().sum(-1) + (twist[:, 3:] / translation_scale_mm).square().sum(-1)


def dispersion(poses, centres, best):
    p = torch.as_tensor(np.asarray(poses), dtype=torch.float64)
    c = torch.as_tensor(centres, dtype=torch.float64)
    frames = torch.stack([relative_frames(x, c) for x in p])
    twists = se3_log(torch.linalg.solve(frames[best][None], frames))
    squared = (twists[..., :3] / np.deg2rad(3.)).square().sum(-1) + (twists[..., 3:] / 3.).square().sum(-1)
    return squared.mean(0).sqrt().numpy()


def perturb(initial, seed):
    # Engineering start perturbations, not patient-motion samples.
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(5, 2, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    angles = directions[:, 0] * np.deg2rad(rng.uniform(0., 5., (5, 1)))
    rotations = Rotation.from_rotvec(angles) * Rotation.from_rotvec(initial[:, :3])
    return np.column_stack((rotations.as_rotvec(), initial[:, 3:] + directions[:, 1] * rng.uniform(0., 5., (5, 1))))


def initialize():
    from adaptive_analysis import deterministic_split
    base, base_hash = frozen_config()
    split = deterministic_split(base['cases'])
    source_files = [Path(__file__), Path(__file__).with_name('adaptive_analysis.py'), Path(__file__).with_name('adaptive_siddon.py'), Path(__file__).with_name('calibrate_adaptive.py'), OUT / 'STATISTICAL_EXECUTION_SPEC.md']
    config = dict(version='adaptive_prior_v1', base_config_hash=base_hash, seed=20261025,
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        split=split, lambda_grid=list(GRID), starts=5, repeats=5,
        main_scenarios=['clean'] + [f'combined_s{i}' for i in range(5)],
        primary='development_calibrated_fixed_C_minus_strong_B',
        endpoints=['surface_rms_tre_mm', 'case_failure_gt3mm'],
        inference='joint superiority, each one-sided alpha .025; CT-volume paired bootstrap',
        proxy_gate='AUROC>=.70 and volume-bootstrap lower95CI>.50, >=5 volumes in each class',
        calibration='.5*clean RMS + .5*mean combined RMS; smallest lambda within one SE',
        motion_and_corruption='unchanged synthetic targets for paired continuity; not clinically calibrated',
        clinical_claims=False, patient_linkage='unverified; CT-volume independence assumed',
        thresholds_mm=[2, 3, 4, 5], optimizer=base['BC'], levels=base['levels'])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Experiment configuration changed; use a new version')
    if not path.exists():
        atomic_json(path, config)
        atomic_json(OUT / 'split.json', split)
    return config, fingerprint(config), base, base_hash


def checked_target(directory, expected_base_hash):
    meta = json.loads((directory / 'metadata.json').read_text())
    if meta['config_hash'] != expected_base_hash:
        raise ValueError('Frozen target configuration mismatch')
    if not meta['accepted']:
        return None, meta
    result = {}
    for view in ('AP', 'LAT'):
        array = np.load(directory / (view + '.npy'), allow_pickle=False)
        if not np.isfinite(array).all() or hashlib.sha256(array.tobytes()).hexdigest() != meta['image_hashes'][view]:
            raise ValueError('Target bytes failed integrity validation')
        result[view] = torch.as_tensor(array, device='cuda')
    return result, meta


def target(name, scenario, case, base, base_hash):
    old_scenario = 'moderate' if scenario == 'clean' else 'combined' if scenario == 'combined_s0' else None
    if old_scenario is not None:
        baseline = OLD / 'targets/moderate' / name
        baseline_meta = json.loads((baseline / 'metadata.json').read_text())
        if baseline_meta['config_hash'] != base_hash:
            raise ValueError('Rejected-target config mismatch')
        if not baseline_meta['accepted']:
            return None, baseline_meta, baseline
        directory = OLD / 'targets' / old_scenario / name
        images, meta = checked_target(directory, base_hash)
        return images, meta, directory
    directory = OUT / 'targets' / scenario / name
    if (directory / 'metadata.json').exists():
        images, meta = checked_target(directory, base_hash)
        return images, meta, directory
    clean, meta = checked_target(OLD / 'targets/moderate' / name, base_hash)
    if clean is None:
        return None, meta, directory
    index = base['cases'].index(name)
    generator = ImageProblem(case, {**base, 'sample_step_mm': .5, 'max_samples': 384})
    with torch.no_grad():
        parts = generator.parts(torch.tensor(meta['motion']['params'], dtype=torch.float32, device='cuda'), 256)
    scales = {v: float(np.log(20.) / np.quantile(x.cpu().numpy()[x.cpu().numpy() > 0], .995)) for v, x in clean.items()}
    if scenario.startswith('combined_s'):
        repeat = int(scenario.rsplit('s', 1)[1])
        images, corruption = degrade(clean, parts, seed=base['seed'] + index * 1009 + repeat * 12347,
            photons=20000, gaussian=.003, occlusion=.15, endpoint=0 if (index // 2) % 2 == 0 else 4, scales=scales)
        meta = {**meta, 'scenario': scenario, 'base_scenario': 'moderate', 'repeat': repeat, 'corruption': corruption}
    elif scenario in ('siddon_clean', 'siddon_combined'):
        from adaptive_siddon import render
        transforms = np.asarray(meta['motion']['transforms'])
        with torch.no_grad():
            parts = {v: [render(case['components'][label], projection, transforms[i], image_shape=(256, 256)).float()
                         for i, label in enumerate(sorted(case['components']))]
                     for v, projection in generator.projections.items()}
            images = {v: torch.stack(rows).sum(0) for v, rows in parts.items()}
        meta = {**meta, 'scenario': scenario, 'target_integrator': 'exact_piecewise_constant_Siddon',
                'target_renderer_sha256': sha256(Path(__file__).with_name('adaptive_siddon.py'))}
        if scenario.endswith('combined'):
            images, corruption = degrade(images, parts, seed=base['seed'] + index * 1009, photons=20000,
                gaussian=.003, occlusion=.15, endpoint=0 if (index // 2) % 2 == 0 else 4)
            meta['corruption'] = corruption
    else:
        raise ValueError(f'Unknown scenario {scenario}')
    save_targets(directory, images, meta)
    return checked_target(directory, base_hash) + (directory,)


def refine(problem, initial, weights, base, scales=(3., 3.)):
    pose = torch.as_tensor(initial, dtype=torch.float32, device='cuda')
    weight = torch.as_tensor(weights, dtype=pose.dtype, device=pose.device).expand(4)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Invalid prior weights')
    levels = []
    for level in base['levels']:
        def objective(p):
            return problem.image_loss(p, level) + (weight * edge_penalties(p, problem.centres, *scales)).mean()
        result = optimize_torch(objective, pose, mode='joint', **base['BC'])
        pose = result.pop('pose')
        levels.append(dict(level=level, **plain(result)))
    return pose.detach().cpu().numpy(), dict(levels=levels, iterations=sum(x['iterations'] for x in levels),
        evaluations=sum(x['evaluations'] for x in levels), stop_reason=levels[-1]['stop_reason'],
        converged=levels[-1]['converged'], hit_max_iterations=any(x['hit_max_iterations'] for x in levels))


def record(path, method, problem, case, meta, config_hash, initial=None, weights=0., imported=None, scales=(3., 3.)):
    target_hash = fingerprint(meta)
    input_signature = fingerprint(dict(method=method, case=case['case'], weights=np.broadcast_to(weights, (4,)).tolist(),
        scales=list(scales), initial=None if initial is None else np.asarray(initial).tolist(),
        imported_sha256=None if imported is None else sha256(imported), implementation_sha256=sha256(__file__)))
    if path.exists():
        row = json.loads(path.read_text())
        if row['config_hash'] != config_hash or row['target_hash'] != target_hash or row.get('input_signature') != input_signature:
            raise ValueError(f'Checkpoint mismatch {path}')
        return row
    torch.cuda.synchronize()
    started = time.monotonic()
    if imported is not None:
        old = json.loads(imported.read_text())
        if old['target_hash'] != target_hash or old['config_hash'] != meta['config_hash']:
            raise ValueError('Cannot reuse a result for a different target')
        pose, diagnostics = np.asarray(old['pose']), old['optimization']
    elif method == 'A':
        pose, diagnostics = optimize_A(problem, problem.config)
    elif method == 'Zero':
        pose = np.zeros((5, 6))
        diagnostics = dict(iterations=0, evaluations=0, stop_reason='not_optimized', converged=None, hit_max_iterations=False)
    else:
        pose, diagnostics = refine(problem, initial, weights, problem.config, scales)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not np.isfinite(pose).all():
        raise FloatingPointError('Nonfinite candidate pose; no result is silently discarded')
    report = evaluate(pose, case, 'A' if method == 'A' else 'B')
    extras = extra_metrics(pose, case, method == 'A')
    for row, extra in zip(report['vertebrae'], extras):
        row.update(extra)
    for key in extras[0]:
        report['metrics'][key] = float(np.mean([x[key] for x in extras]))
    local = global_to_local(pose, case['centres']) if method == 'A' else pose
    with torch.no_grad():
        p = torch.tensor(local, dtype=torch.float32, device='cuda')
        penalties = edge_penalties(p, problem.centres, *scales).cpu().numpy()
        image_loss = float(problem.image_loss(p, 256))
    if not np.isfinite(image_loss) or not np.isfinite(penalties).all():
        raise FloatingPointError('Nonfinite objective; explicit failed execution')
    weighted = float(np.mean(np.broadcast_to(weights, (4,)) * penalties))
    result = dict(case=case['case'], scenario=path.parent.parent.name, method=method,
        config_hash=config_hash, target_hash=target_hash, input_signature=input_signature, pose=pose.tolist(),
        runtime_seconds=elapsed, ncc_loss=image_loss, motion_penalty=float(penalties.mean()),
        edge_penalties=penalties.tolist(), edge_weights=np.broadcast_to(weights, (4,)).tolist(),
        weighted_prior=weighted, weighted_fraction=weighted / max(image_loss + weighted, 1e-20),
        scales=list(scales), optimization=diagnostics, **report, implementation_sha256=sha256(__file__))
    if imported:
        result['imported_provenance'] = dict(path=str(imported.relative_to(ROOT)), sha256=sha256(imported),
                                            original_runtime_seconds=old['runtime_seconds'])
    atomic_json(path, plain(result))
    print(json.dumps(dict(case=case['case'], scenario=result['scenario'], method=method,
        seconds=round(elapsed, 2), rms=round(report['metrics']['surface_rms_tre_mm'], 5), stop=diagnostics['stop_reason'])), flush=True)
    return result


def run(name, scenarios, weights, config, config_hash, base, base_hash, adaptive=None):
    case = verified_case(name)
    problem = ImageProblem(case, base)
    index = base['cases'].index(name)
    for scenario in scenarios:
        images, meta, directory = target(name, scenario, case, base, base_hash)
        if images is None:
            atomic_json(OUT / 'eligibility' / scenario / (name + '.json'),
                        dict(case=name, accepted=False, source=str(directory), motion=meta['motion']))
            continue
        atomic_json(OUT / 'eligibility' / scenario / (name + '.json'),
                    dict(case=name, accepted=True, target_hash=fingerprint(meta), source=str(directory)))
        case['gt_params'] = np.asarray(meta['motion']['params'])
        case['gt_matrices'] = np.asarray(meta['motion']['transforms'])
        problem.set_targets(images)
        result_dir, bank_dir = OUT / 'results' / scenario / name, OUT / 'banks' / scenario / name
        old_scenario = 'moderate' if scenario == 'clean' else 'combined' if scenario == 'combined_s0' else None
        old_a = OLD / 'results' / old_scenario / name / 'A.json' if old_scenario else None
        a = record(result_dir / 'A.json', 'A', problem, case, meta, config_hash, imported=old_a)
        record(result_dir / 'Zero.json', 'Zero', problem, case, meta, config_hash)
        initial = global_to_local(np.asarray(a['pose']), case['centres'])
        candidates = []
        for k in range(config['starts']):
            old_b = OLD / 'results' / old_scenario / name / 'B.json' if old_scenario and k == 0 else None
            start = initial if k == 0 else perturb(initial, config['seed'] + index * 1009 + k * 7919)
            candidates.append(record(bank_dir / f'start_{k}.json', f'start_{k}', problem, case, meta, config_hash,
                                     initial=start, imported=old_b))
        if not np.isfinite([x['ncc_loss'] for x in candidates]).all():
            raise FloatingPointError('Candidate bank contains nonfinite objectives')
        winner = int(np.argmin([x['ncc_loss'] for x in candidates]))
        u = dispersion([x['pose'] for x in candidates], case['centres'], winner)
        atomic_json(bank_dir / 'selection.json', dict(case=name, scenario=scenario, best=winner,
            proxy_edges=u.tolist(), proxy_case=float(u.max()), config_hash=config_hash, target_hash=fingerprint(meta),
            ncc_losses=[x['ncc_loss'] for x in candidates],
            candidate_runtime_seconds=sum(x.get('imported_provenance', {}).get('original_runtime_seconds', x['runtime_seconds']) for x in candidates)))
        best = np.asarray(candidates[winner]['pose'])
        record(result_dir / 'B.json', 'B', problem, case, meta, config_hash, initial=best)
        for weight in weights:
            if weight:
                method = f'lambda_{weight:g}'
                record(result_dir / (method + '.json'), method, problem, case, meta, config_hash, initial=best, weights=weight)
        if adaptive is not None:
            edge_weights = adaptive['lambda_max'] * u / (u + adaptive['tau'])
            record(result_dir / 'C_adaptive.json', 'C_adaptive', problem, case, meta, config_hash,
                   initial=best, weights=edge_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'develop', 'evaluate', 'sensitivity'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case')
    parser.add_argument('--scenarios', nargs='+')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    torch.set_num_threads(1)
    config, config_hash, base, base_hash = initialize()
    if args.action == 'init':
        print(json.dumps(config, indent=2)); return
    lock = (OUT / f'{args.action}_{args.shard}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = 'development' if args.action == 'develop' else 'evaluation'
    names = config['split'][key]
    if args.action == 'sensitivity':
        names = names[:10]
    names = names[args.shard::args.shards]
    if args.case:
        if args.case not in config['split'][key]:
            raise ValueError('Case belongs to the other split')
        names = [args.case]
    adaptive = None
    if args.action == 'develop':
        weights = GRID
    else:
        selection = json.loads((OUT / 'calibration/selection.json').read_text())
        if selection['config_hash'] != config_hash:
            raise ValueError('Calibration configuration mismatch')
        weights = sorted(set([selection['fixed_lambda'], .1]))
        adaptive = selection.get('adaptive')
    scenarios = args.scenarios or (['siddon_clean', 'siddon_combined'] if args.action == 'sensitivity' else config['main_scenarios'])
    for name in names:
        run(name, scenarios, weights, config, config_hash, base, base_hash, adaptive)
    atomic_json(OUT / f'{args.action}_{args.shard}_completion.json',
                dict(config_hash=config_hash, cases=names, scenarios=scenarios, status='COMPLETE'))


if __name__ == '__main__':
    main()
