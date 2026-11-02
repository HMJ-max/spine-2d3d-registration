"""Supplementary engineering sensitivity on the first 10 evaluation CT volumes.

Published means describe standing-to-prone endplate changes, NOT supine-to-prone
motion distributions. Centroid hinges and noise severities are engineering proxies.
No outcome selects a case, profile, severity, prior scale, or lambda here.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F
import adaptive_study as study
from final_study_support import articulated, degrade, sha256
from generate_posture_dataset import field_of_view, plain
from posture_geometry import prepare_geometry
from spine_registration.benchmark import ImageProblem, atomic_json, fingerprint

JOINTS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5']
PUBLISHED = {'published_jackson_mean': 'miyazaki_jackson_mean',
             'published_four_poster_mean': 'miyazaki_four_poster_mean'}
MOTIONS = tuple(PUBLISHED) + ('reverse_moderate', 'upper_dominant_8deg', 'lower_dominant_8deg')
CORRUPTIONS = {'combined_low': dict(photons=40000, gaussian=.0015, occlusion=.075),
               'combined_high': dict(photons=10000, gaussian=.006, occlusion=.30)}
AMPLITUDES = {'small_2deg': 'small', 'stress_16deg': 'large'}
SCENARIOS = MOTIONS + tuple(CORRUPTIONS) + tuple(AMPLITUDES)
PRIOR_SCALES = ((3., 1.5), (3., 6.))


def assemble_angles(evidence, original_angles):
    """Use published mean deltas verbatim; only engineering patterns use case sign."""
    if evidence['joint_order'] != JOINTS:
        raise ValueError('Published joint order must be L1/L2 through L4/L5')
    original = np.asarray(original_angles, dtype=float)
    if original.shape != (4,) or not np.isfinite(original).all() or original.sum() == 0:
        raise ValueError('Four finite, nonzero-total original moderate angles required')
    profiles = {row['id']: row for row in evidence['profiles']}
    angles = {}
    for scenario, profile_id in PUBLISHED.items():
        row = profiles[profile_id]
        delta = np.asarray(row['delta_deg'], dtype=float)
        difference = np.asarray(row['prone_mean_deg']) - np.asarray(row['standing_mean_deg'])
        if (delta.shape != (4,) or not np.isfinite(delta).all()
                or not np.allclose(delta, difference, atol=1e-8, rtol=0)):
            raise ValueError('Published delta must equal four prone minus standing means')
        angles[scenario] = delta
    sign = np.sign(original.sum())
    angles['reverse_moderate'] = -original
    angles['upper_dominant_8deg'] = sign * 8 * np.array([.4, .3, .2, .1])
    angles['lower_dominant_8deg'] = sign * 8 * np.array([.1, .2, .3, .4])
    return angles


def original_global_motion(metadata, centres):
    """Recover rotation and displacement about L5, not the matrix translation."""
    matrix = np.asarray(metadata['motion']['global_transform'], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        raise ValueError('Original global transform must be rigid and finite')
    pivot = np.asarray(centres)[-1]
    return Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, :3] @ pivot + matrix[:3, 3] - pivot


def freeze():
    """Require development selection; freeze provenance before target generation."""
    config, config_hash, base, base_hash = study.initialize()
    split_path = study.OUT / 'split.json'
    split = json.loads(split_path.read_text())
    if split != config['split']:
        raise ValueError('Saved split differs from frozen main-study split')
    selection_path = study.OUT / 'calibration/selection.json'
    selection = json.loads(selection_path.read_text())
    if selection['config_hash'] != config_hash or selection['fixed_lambda'] not in study.GRID:
        raise ValueError('Calibration does not match the frozen configuration')
    evidence_path = study.OUT / 'evidence/CLINICAL_SEGMENT_PROFILES.json'
    evidence = json.loads(evidence_path.read_text())
    published = assemble_angles(evidence, [2.] * 4)
    sources = [Path(__file__), Path(study.__file__)] + [Path(__file__).with_name(name) for name in
        ['final_study_support.py', 'posture_geometry.py', 'generate_posture_dataset.py']]
    manifest = dict(version='adaptive_robustness_v1', config_hash=config_hash,
        base_config_hash=base_hash, cases=split['evaluation'][:10], split_sha256=sha256(split_path),
        selected_lambda=selection['fixed_lambda'], selection_sha256=sha256(selection_path),
        source_hashes={str(p.relative_to(study.ROOT)): sha256(p) for p in sources},
        evidence_sha256=sha256(evidence_path), source_doi=evidence['source_doi'],
        published_angles_deg={k: published[k].tolist() for k in PUBLISHED}, joint_order=JOINTS,
        motion_scenarios=list(MOTIONS), corruption_scenarios=CORRUPTIONS,
        amplitude_scenarios=AMPLITUDES,
        amplitude_caveat='Frozen synthetic 2/16-degree endpoint conditions; reuse original images, actual joint angles, global motion and eligibility without replacements or clinical interpretation',
        fixed_comparison_weights=sorted(set([selection['fixed_lambda'], .1])),
        prior_scales=list(map(list, PRIOR_SCALES)), prior_scale_scenarios=['clean', 'combined_s0'],
        upper_weights=[.4, .3, .2, .1], lower_weights=[.1, .2, .3, .4], total_engineering_angle_deg=8.,
        global_motion='New motion patterns retain original moderate placement about L5; amplitude conditions retain their own complete original motion',
        profile_sign='Positive RAS-x cranial-relative rotation increases sagittal lordotic frame turning; measured endplate deltas embedded with that convention',
        target_rendering=dict(resolution=512, resize=256, resize_mode='area',
            sample_step_mm=base['target_sample_step_mm'], max_samples=base['target_max_samples']),
        acceptance='Unchanged articulated() geometry acceptance and field_of_view(); retain every rejection',
        selection_rule='First 10 saved evaluation IDs before supplementary evaluation; no replacements',
        interpretation='Supplementary engineering sensitivity, not population or clinical calibration',
        published_profile_caveats=evidence['limitations'],
        corruption_caveat='Photon, Gaussian and occlusion levels are uncalibrated engineering settings',
        scale_caveat='Relative translation/rotation weighting changes; fixed development lambda. Lambda zero makes scales inactive.')
    destination = study.OUT / 'robustness'
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'manifest.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = destination / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Supplementary source/config changed; use a new output version')
        if not path.exists():
            atomic_json(path, manifest)
    return manifest, evidence, selection, config, config_hash, base, base_hash


def checked_prepared(name, scenario, manifest, base_hash):
    directory = study.OUT / 'targets' / scenario / name
    metadata = json.loads((directory / 'metadata.json').read_text())
    if (metadata.get('robustness_manifest_hash') != fingerprint(manifest)
            or metadata.get('base_config_hash') != base_hash):
        raise ValueError(f'Supplementary target provenance mismatch: {directory}')
    return study.checked_target(directory, base_hash)


def prepare(name, scenarios, manifest, evidence, base, base_hash):
    case = study.verified_case(name)
    original_dir = study.OLD / 'targets/moderate' / name
    original = json.loads((original_dir / 'metadata.json').read_text())
    if original['config_hash'] != base_hash:
        raise ValueError('Original moderate target configuration mismatch')
    angles = assemble_angles(evidence, original['motion']['segment_angles_deg'])
    rotation, translation = original_global_motion(original, case['centres'])
    generator = ImageProblem(case, {**base, 'sample_step_mm': base['target_sample_step_mm'],
                                    'max_samples': base['target_max_samples']})
    prepared = None
    for scenario in scenarios:
        directory = study.OUT / 'targets' / scenario / name
        if (directory / 'metadata.json').exists():
            checked_prepared(name, scenario, manifest, base_hash)
            continue
        metadata = dict(case=name, scenario=scenario, config_hash=base_hash, base_config_hash=base_hash,
            adaptive_config_hash=manifest['config_hash'], robustness_manifest_hash=fingerprint(manifest),
            original_moderate_target_hash=fingerprint(original), centres=case['centres'].tolist(),
            interpretation=manifest['interpretation'])
        if scenario in MOTIONS:
            if prepared is None:
                with np.load(study.ROOT / 'inputs/ppcnet75' / name / 'centred_case.npz', allow_pickle=False) as archive:
                    prepared = prepare_geometry(archive['segmentation'], archive['affine_mm'])
                if not np.allclose(prepared['centres'], case['centres'], atol=1e-6):
                    raise ValueError('Prepared and frozen case centroids differ')
            motion = articulated(prepared, angles[scenario], rotation, translation)
            motion['field_of_view'] = field_of_view(case, motion, generator.projections)
            motion['accepted'] &= bool(motion['field_of_view']['accepted'])
            metadata.update(motion=plain(motion), accepted=motion['accepted'],
                            target_rendering=manifest['target_rendering'])
            if scenario in PUBLISHED:
                metadata.update(profile_id=PUBLISHED[scenario], source_comparison=evidence['comparison'],
                                clinical_calibration=False, profile_caveats=evidence['limitations'])
            if motion['accepted']:
                with torch.no_grad():
                    parts = generator.parts(torch.tensor(motion['params'], dtype=torch.float32, device='cuda'), 512)
                    images = {v: F.interpolate(torch.stack(rows).sum(0)[None, None],
                                               size=(256, 256), mode='area')[0, 0] for v, rows in parts.items()}
        elif scenario in AMPLITUDES:
            source = study.OLD / 'targets' / AMPLITUDES[scenario] / name
            images, amplitude = study.checked_target(source, base_hash)
            metadata.update(motion=amplitude['motion'], accepted=amplitude['accepted'],
                base_scenario=AMPLITUDES[scenario], source_target=str(source.relative_to(study.ROOT)),
                original_amplitude_target_hash=fingerprint(amplitude),
                clinical_calibration=False, amplitude_caveat=manifest['amplitude_caveat'],
                target_rendering='Original frozen target bytes reused without rendering')
        else:
            clean, _ = study.checked_target(original_dir, base_hash)
            metadata.update(motion=original['motion'], accepted=original['accepted'], base_scenario='moderate',
                            corruption_settings=CORRUPTIONS[scenario], clinical_calibration=False,
                            corruption_caveat=manifest['corruption_caveat'])
            if clean is not None:
                index = base['cases'].index(name)
                with torch.no_grad():
                    parts = generator.parts(torch.tensor(original['motion']['params'], dtype=torch.float32, device='cuda'), 256)
                images, corruption = degrade(clean, parts, seed=base['seed'] + index * 1009,
                    endpoint=0 if (index // 2) % 2 == 0 else 4, **CORRUPTIONS[scenario])
                metadata['corruption'] = corruption
        if metadata['accepted']:
            study.save_targets(directory, images, metadata)
        else:
            atomic_json(directory / 'metadata.json', plain(metadata))
        print(json.dumps(dict(case=name, scenario=scenario, accepted=metadata['accepted'])), flush=True)


def run_scales(name, manifest, config_hash, base, base_hash):
    case = study.verified_case(name)
    problem = ImageProblem(case, base)
    for scenario in manifest['prior_scale_scenarios']:
        images, meta, _ = study.target(name, scenario, case, base, base_hash)
        if images is None:
            atomic_json(study.OUT / 'robustness/scale_eligibility' / scenario / (name + '.json'),
                        dict(case=name, accepted=False, target_hash=fingerprint(meta)))
            continue
        bank = study.OUT / 'banks' / scenario / name
        selection = json.loads((bank / 'selection.json').read_text())
        winner = json.loads((bank / f"start_{selection['best']}.json").read_text())
        for row in (selection, winner):
            if row['config_hash'] != config_hash or row['target_hash'] != fingerprint(meta):
                raise ValueError('Existing bank winner belongs to another configuration or target')
        case['gt_params'], case['gt_matrices'] = np.asarray(meta['motion']['params']), np.asarray(meta['motion']['transforms'])
        problem.set_targets(images)
        for scales in PRIOR_SCALES:
            method = f'C_scale_t{scales[1]:g}'
            study.record(study.OUT / 'results' / scenario / name / (method + '.json'), method,
                problem, case, meta, config_hash, initial=np.asarray(winner['pose']),
                weights=manifest['selected_lambda'], scales=scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'scales'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    if args.action == 'scales' and args.scenarios:
        parser.error('Prior scales use only frozen clean and combined_s0 scenarios')
    torch.set_num_threads(1)
    manifest, evidence, selection, config, config_hash, base, base_hash = freeze()
    if args.case and args.case not in manifest['cases']:
        parser.error('Case must belong to the fixed first 10 evaluation IDs')
    names = [args.case] if args.case else manifest['cases'][args.shard::args.shards]
    scenarios = args.scenarios or list(SCENARIOS)
    with (study.OUT / 'robustness' / f'{args.action}_{args.shard}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names:
            if args.action == 'prepare':
                prepare(name, scenarios, manifest, evidence, base, base_hash)
            elif args.action == 'run':
                for scenario in scenarios:
                    checked_prepared(name, scenario, manifest, base_hash)
                study.run(name, scenarios, manifest['fixed_comparison_weights'], config, config_hash,
                          base, base_hash, selection.get('adaptive'))
            else:
                run_scales(name, manifest, config_hash, base, base_hash)
        atomic_json(study.OUT / 'robustness' / f'{args.action}_{args.shard}_completion.json',
            dict(robustness_manifest_hash=fingerprint(manifest), cases=names, scenarios=(
                manifest['prior_scale_scenarios'] if args.action == 'scales' else scenarios), status='COMPLETE'))


if __name__ == '__main__':
    main()
