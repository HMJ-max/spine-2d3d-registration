"""Generate literature-informed articulated posture targets; never registration results."""
from __future__ import annotations

import argparse
import csv
import portable_lock as fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from spine_registration.benchmark import (LABELS, ImageProblem, atomic_json, evaluate,
    fingerprint, load_case, projections)
from posture_geometry import make_articulated_motion, prepare_geometry, audit_geometry

CONDITIONS = {'near_preserved': [-2., 2.], 'extension_scenario': [3., 8.], 'flexion_stress': [-8., -3.]}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def config_for(manifest):
    files = sorted((ROOT / 'code/src/spine_registration').glob('*.py'))
    files += [Path(__file__), Path(__file__).with_name('posture_geometry.py')]
    return {'version': 1, 'seed': 20261023, 'cases': [x['case'] for x in manifest['cases']],
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in files},
        'input_manifest_hash': fingerprint(manifest), 'conditions_total_rx_deg': CONDITIONS,
        'segment_weight_uniform': [.8, 1.2], 'global_rotvec_axis_uniform_deg': [-3., 3.],
        'global_translation_axis_uniform_mm': [-5., 5.], 'global_pivot': 'L5 centroid',
        'hinge': 'adjacent full-vertebra centroid midpoint; not measured disc centre',
        'interpretation': 'Synthetic bounded scenarios, not clinical population distributions or measured Cobb changes',
        'coordinate_frame': 'centred CT RAS mm; rotations are axis-angle, not Euler',
        'global_pose_shared_across_conditions': True, 'max_candidate_bundles': 64,
        'acceptance': {'centroid_distance_ratio': [.9, 1.1], 'deep_penetration_fraction': 0.,
            'max_NN_overlap_fraction_smaller_volume': .01, 'detector_margin_pixels': 6,
            'rule': 'Geometry only; reject entire three-condition bundle, never use registration error or motion penalty'},
        'levels': [256], 'pixel_spacing_mm': 2., 'source_to_isocenter_mm': 750.,
        'source_to_detector_mm': 1200., 'sample_step_mm': 1., 'max_samples': 192, 'chunk_rows': 64,
        'intensity': 'same frozen bone-selective linear ray integral as abc_revision; no log1p',
        'view_convention': 'Legacy AP label: source at negative RAS y (physically PA); LAT source at negative x'}


def sagittal_curve_proxy(centres):
    """Sagittal centreline turning angle, not an endplate-based Cobb angle."""
    cranial, caudal = centres[0] - centres[1], centres[3] - centres[4]
    a = np.arctan2(caudal[1], caudal[2]) - np.arctan2(cranial[1], cranial[2])
    return float(np.rad2deg(np.arctan2(np.sin(a), np.cos(a))))


def sample_bundle(centres, rng):
    rotation = np.deg2rad(rng.uniform(-3, 3, 3))
    translation = rng.uniform(-5, 5, 3)
    bundle = {}
    for condition, limits in CONDITIONS.items():
        total = rng.uniform(*limits)
        weights = rng.uniform(.8, 1.2, 4)
        angles = total * weights / weights.sum()
        bundle[condition] = make_articulated_motion(centres, angles, rotation, translation)
        bundle[condition]['segment_angles_deg'] = angles
        bundle[condition]['total_relative_rx_deg'] = total
        articulated = bundle[condition]['articulated_transforms']
        moved = np.einsum('nij,nj->ni', articulated[:, :3, :3], centres) + articulated[:, :3, 3]
        before, after = sagittal_curve_proxy(centres), sagittal_curve_proxy(moved)
        change = (after - before + 180) % 360 - 180
        bundle[condition]['sagittal_centreline_proxy_deg'] = {'before': before, 'after': after, 'change': change}
        if change * total < -1e-8:
            raise ValueError('Requested bending sign conflicts with sagittal centreline proxy')
    return bundle


def field_of_view(case, motion, cameras):
    details = {}
    for view, projection in cameras.items():
        image_points = []
        for corners, transform in zip(case['corners'], motion['transforms']):
            moved = corners @ transform[:3, :3].T + transform[:3, 3]
            image_points.append(np.column_stack([moved, np.ones(len(moved))]) @ projection.T)
        projected = np.concatenate(image_points)
        if (projected[:, 2] <= 0).any():
            return {'accepted': False, 'reason': 'nonpositive_camera_depth'}
        pixels = projected[:, :2] / projected[:, 2:]
        details[view] = {'min_uv': pixels.min(0).tolist(), 'max_uv': pixels.max(0).tolist(),
                         'min_depth_mm': float(projected[:, 2].min())}
    accepted = all(min(d['min_uv']) >= 6 and max(d['max_uv']) <= 249 for d in details.values())
    return {'accepted': accepted, 'views': details}


def geometry_reasons(qa, fov):
    reasons = []
    if not fov['accepted']:
        reasons.append('detector_FOV')
    if qa['max_penetration_fraction_beyond_tolerance'] > 0:
        reasons.append('penetration_beyond_voxel_tolerance')
    if qa['max_overlap_fraction_smaller_volume'] is not None and qa['max_overlap_fraction_smaller_volume'] > .01:
        reasons.append('overlap_fraction_above_1_percent')
    if any(not .9 <= edge['centroid_distance_ratio'] <= 1.1 for edge in qa['edges']):
        reasons.append('centroid_spacing')
    if qa['max_hinge_residual_mm'] is not None and qa['max_hinge_residual_mm'] > 1e-8:
        reasons.append('hinge_discontinuity')
    return reasons


def verify_record(path, config_hash, expected_input, expected_case, expected_condition):
    record = json.loads(path.read_text())
    digest = record.pop('record_hash')
    if digest != fingerprint(record) or record['config_hash'] != config_hash or record['input_hashes'] != expected_input:
        raise ValueError(f'Incompatible checkpoint: {path}')
    if (record['case'], record['condition']) != (expected_case, expected_condition):
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    if set(record['artifacts_sha256']) != {'centred_case.npz', 'motion_ground_truth.json', 'AP.npy', 'LAT.npy'}:
        raise ValueError(f'Incomplete checkpoint artifacts: {path}')
    for name, expected in record['artifacts_sha256'].items():
        if sha256(path.parent / name) != expected:
            raise ValueError(f'Corrupt dataset artifact: {path.parent / name}')
    return record


def write_array(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, value)
    temporary.replace(path)


def run_case(row, inputs, output, config):
    case_id = row['case']
    source = inputs / case_id
    expected = {'centred_case.npz': row['volume_sha256'], 'motion_ground_truth.json': row['gt_sha256']}
    if any(sha256(source / name) != digest for name, digest in expected.items()):
        raise ValueError(f'Changed original input: {case_id}')
    config_hash = fingerprint(config)
    paths = {c: output / 'dataset' / c / case_id for c in CONDITIONS}
    completed = {c for c, p in paths.items() if (p / 'record.json').exists()}
    for condition in completed:
        verify_record(paths[condition] / 'record.json', config_hash, expected, case_id, condition)
    if len(completed) == len(CONDITIONS):
        print(f'resume {case_id} all 3 conditions', flush=True)
        return
    case = load_case(source)
    with np.load(source / 'centred_case.npz', allow_pickle=False) as archive:
        geometry = prepare_geometry(archive['segmentation'], archive['affine_mm'])
    if not np.allclose(geometry['centres'], case['centres'], atol=1e-8):
        raise ValueError('Geometry/renderer centroid mismatch')
    baseline = audit_geometry(geometry, np.tile(np.eye(4), (5, 1, 1)), include_overlap=True)
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], int(case_id.split('_')[-1])]))
    attempts, chosen = [], None
    for attempt in range(1, config['max_candidate_bundles'] + 1):
        bundle = sample_bundle(case['centres'], rng)
        checks, rejected = {}, {}
        for condition, motion in bundle.items():
            qa = audit_geometry(geometry, motion['transforms'], motion['pivots'], include_overlap=True)
            fov = field_of_view(case, motion, projections(config['pixel_spacing_mm']))
            reasons = geometry_reasons(qa, fov)
            checks[condition] = {'geometry': qa, 'field_of_view': fov}
            if reasons:
                rejected[condition] = reasons
        attempts.append({'attempt': attempt, 'total_relative_rx_deg': {k: v['total_relative_rx_deg'] for k, v in bundle.items()},
                         'rejected': rejected})
        if not rejected:
            chosen = bundle
            break
    if chosen is None:
        atomic_json(output / 'rejected' / f'{case_id}.json', plain({'case': case_id, 'attempts': attempts}))
        raise RuntimeError(f'{case_id}: no geometry-valid bundle in prespecified attempts; do not silently relax limits')
    problem = ImageProblem(case, config)
    original = json.loads((source / 'motion_ground_truth.json').read_text())
    for condition, motion in chosen.items():
        if condition in completed:
            continue
        directory = paths[condition]
        directory.mkdir(parents=True, exist_ok=True)
        cached_volume = directory / 'centred_case.npz'
        if not cached_volume.exists():
            cached_volume.symlink_to(source.resolve() / 'centred_case.npz')
        targets, _ = problem.synthesize(motion['params'])
        problem.set_targets(targets)
        with torch.no_grad():
            gt_loss = float(problem.image_loss(torch.as_tensor(motion['params'], device='cuda', dtype=torch.float32), 256))
            zero_loss = float(problem.image_loss(torch.zeros((5, 6), device='cuda'), 256))
        if abs(gt_loss) > 2e-6:
            raise RuntimeError(f'Forward model mismatch: {gt_loss}')
        truth = json.loads(json.dumps(original))
        truth['motions'] = {str(label): dict(zip(['rx_deg', 'ry_deg', 'rz_deg', 'tx_mm', 'ty_mm', 'tz_mm'],
                                      [*np.rad2deg(p[:3]), *p[3:]])) for label, p in zip(LABELS, motion['params'])}
        truth['transforms_mm'] = {str(label): m.tolist() for label, m in zip(LABELS, motion['transforms'])}
        truth['motion_parameter_units']['rotation_representation'] = 'axis-angle vector components in degrees, not Euler angles'
        truth['posture_generation'] = plain({'condition': condition, 'config_hash': config_hash,
            'accepted_attempt': attempt, **motion})
        atomic_json(directory / 'motion_ground_truth.json', plain(truth))
        for view, target in targets.items():
            write_array(directory / f'{view}.npy', target.cpu().numpy())
        changed_case = {**case, 'gt_params': motion['params'], 'gt_matrices': motion['transforms']}
        zero = evaluate(np.zeros((5, 6)), changed_case, 'Zero')
        # Removing only G measures articulated deformation without placement error.
        local_case = {**case, 'gt_matrices': motion['articulated_transforms']}
        local_params = np.zeros((5, 6))
        local_params[:, :3] = Rotation.from_matrix(motion['articulated_transforms'][:, :3, :3]).as_rotvec()
        local_params[:, 3:] = np.einsum('nij,nj->ni', motion['articulated_transforms'][:, :3, :3], case['centres']) + motion['articulated_transforms'][:, :3, 3] - case['centres']
        local_case['gt_params'] = local_params
        deformation = evaluate(np.zeros((5, 6)), local_case, 'Zero')
        record = plain({'case': case_id, 'condition': condition, 'config_hash': config_hash,
            'input_hashes': expected, 'accepted_attempt': attempt, 'attempts': attempts,
            'motion': motion, 'baseline_geometry': baseline, **checks[condition],
            'zero_motion': zero, 'articulated_only_zero_motion': deformation,
            'gt_image_loss': gt_loss, 'zero_image_loss': zero_loss,
            'projection_matrices': problem.projections,
            'artifacts_sha256': {name: sha256(directory / name) for name in ['centred_case.npz', 'motion_ground_truth.json', 'AP.npy', 'LAT.npy']}})
        record['record_hash'] = fingerprint(record)
        atomic_json(directory / 'record.json', record)
        print(f"done {case_id} {condition} attempt={attempt} total_rx={motion['total_relative_rx_deg']:.2f} deg zero_T={zero['metrics']['translation_error_mm']:.2f} mm", flush=True)
    del problem
    torch.cuda.empty_cache()


def summarize(output, config, manifest):
    rows = []
    inputs = {r['case']: {'centred_case.npz': r['volume_sha256'], 'motion_ground_truth.json': r['gt_sha256']} for r in manifest['cases']}
    for condition in CONDITIONS:
        for case in config['cases']:
            p = output / 'dataset' / condition / case / 'record.json'
            if not p.exists():
                continue
            r = verify_record(p, fingerprint(config), inputs[case], case, condition)
            rows.append({'case': case, 'condition': condition, 'attempts': r['accepted_attempt'],
                'total_relative_rx_deg': r['motion']['total_relative_rx_deg'],
                'zero_translation_mm': r['zero_motion']['metrics']['translation_error_mm'],
                'zero_rotation_deg': r['zero_motion']['metrics']['rotation_error_deg'],
                'deformation_only_translation_mm': r['articulated_only_zero_motion']['metrics']['translation_error_mm'],
                'deformation_only_rotation_deg': r['articulated_only_zero_motion']['metrics']['rotation_error_deg'],
                'max_overlap_fraction': r['geometry']['max_overlap_fraction_smaller_volume'],
                'max_depth_beyond_tolerance_mm': r['geometry']['max_excess_depth_mm'],
                'gt_image_loss': r['gt_image_loss'], 'path': str(p.parent.relative_to(output))})
    summary = {'status': 'COMPLETE' if len(rows) == len(config['cases']) * 3 else 'PARTIAL',
        'subjects': len(config['cases']), 'expected_scenarios': len(config['cases']) * 3,
        'completed_scenarios': len(rows), 'config_hash': fingerprint(config), 'by_condition': {}}
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        if selected:
            summary['by_condition'][condition] = {key: {'mean': float(np.mean([r[key] for r in selected])),
                'min': float(min(r[key] for r in selected)), 'max': float(max(r[key] for r in selected))}
                for key in ['total_relative_rx_deg', 'zero_translation_mm', 'zero_rotation_deg',
                            'deformation_only_translation_mm', 'deformation_only_rotation_deg', 'max_overlap_fraction']}
    atomic_json(output / 'summary.json', summary)
    if rows:
        with (output / 'motion_summary.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs/ppcnet75')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/posture_v1')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    torch.set_num_threads(4)
    if args.cases and len(set(args.cases)) != len(args.cases):
        parser.error('Duplicate case identifiers')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'generation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    config = config_for(manifest)
    if args.cases and not set(args.cases) <= set(config['cases']):
        parser.error('Unknown case identifier')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing output uses a different protocol/source; use a new versioned directory')
    atomic_json(path, config)
    for row in manifest['cases']:
        if not args.cases or row['case'] in args.cases:
            run_case(row, args.inputs, args.output, config)
    print(json.dumps(summarize(args.output, config, manifest), indent=2), flush=True)


if __name__ == '__main__':
    main()
