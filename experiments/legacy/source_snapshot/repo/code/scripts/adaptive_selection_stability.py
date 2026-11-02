"""Descriptive development-bootstrap stability; never changes selected lambda."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_analysis import BOOTSTRAP_SEED, LAMBDA_GRID, select_lambda
from calibrate_adaptive import OUT, load_development
from spine_registration.benchmark import atomic_json, fingerprint


def bootstrap_selection_frequencies(clean, combined, *, bootstrap_samples=10000):
    """Resample whole development-volume rows and rerun the unchanged locked rule."""
    clean, combined = np.asarray(clean, float), np.asarray(combined, float)
    point = select_lambda(clean, combined)  # Validates finite risks, grid and five seeds.
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError('Require at least 100 bootstrap samples; study report uses 10000')
    selected_counts = np.zeros(len(LAMBDA_GRID), dtype=int)
    argmin_counts = np.zeros(len(LAMBDA_GRID), dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(clean), len(clean))
        choice = select_lambda(clean[indices], combined[indices])
        selected_counts[LAMBDA_GRID.index(choice['selected_lambda'])] += 1
        argmin_counts[LAMBDA_GRID.index(choice['minimum_risk_lambda'])] += 1
    return dict(bootstrap_samples=bootstrap_samples, bootstrap_seed=BOOTSTRAP_SEED,
        n_development_volumes=len(clean), original_selection=point,
        frequencies=[dict(lambda_value=weight,
            smallest_within_one_se_count=int(selected_counts[i]),
            smallest_within_one_se_frequency=float(selected_counts[i] / bootstrap_samples),
            argmin_count=int(argmin_counts[i]), argmin_frequency=float(argmin_counts[i] / bootstrap_samples))
            for i, weight in enumerate(LAMBDA_GRID)])


def write_selection_stability(output_root=OUT, *, bootstrap_samples=10000):
    root = Path(output_root)
    data = load_development(root)
    fixed_path = root / 'calibration/fixed_selection.json'
    fixed_bytes = fixed_path.read_bytes()
    fixed = json.loads(fixed_bytes)
    point = select_lambda(data['clean'], data['combined'])
    if (fixed.get('status') != 'FIXED_CALIBRATION_COMPLETE' or fixed.get('config_hash') != data['config_hash']
            or fixed.get('included_cases') != data['included'] or fixed.get('excluded_cases') != data['excluded']
            or fixed.get('selection') != point or fixed.get('fixed_lambda') != point['selected_lambda']
            or fixed.get('provenance') != data['provenance'] or len(fixed.get('folds', [])) != 5):
        raise ValueError('Frozen fixed selection does not match validated development inputs')
    folds, fold_counts = [], dict.fromkeys(LAMBDA_GRID, 0)
    for index, (saved, configured) in enumerate(zip(fixed['folds'], data['config']['split']['development_folds'])):
        held = np.array([case in configured for case in data['included']])
        training = np.asarray(data['included'])[~held].tolist()
        validation = np.asarray(data['included'])[held].tolist()
        if not validation or len(training) < 2:
            raise ValueError('Incomplete development fold for stability reporting')
        selected = select_lambda(data['clean'][~held], data['combined'][~held])['selected_lambda']
        if (saved.get('fold') != index or saved.get('training_cases') != training
                or saved.get('validation_cases') != validation or saved.get('selected_lambda') != selected):
            raise ValueError('Saved fold selection is not the training-only locked-rule selection')
        folds.append(dict(fold=index, selected_lambda=selected,
                          n_training=len(training), n_validation=len(validation)))
        fold_counts[selected] += 1
    result = bootstrap_selection_frequencies(data['clean'], data['combined'], bootstrap_samples=bootstrap_samples)
    for row in result['frequencies']:
        count = fold_counts[row['lambda_value']]
        row.update(training_only_fold_count=count, training_only_fold_frequency=count / 5)
    result.update(status='DESCRIPTIVE_DEVELOPMENT_SELECTION_STABILITY', config_hash=data['config_hash'],
        original_fixed_lambda=fixed['fixed_lambda'], included_cases=data['included'], excluded_cases=data['excluded'],
        training_only_fold_selections=folds, resampling_unit='independent CT volume; all five seeds travel together',
        rule='Smallest lambda within one SE of minimum risk: nonstandard motion-preserving tolerance heuristic; classical glmnet one-SE favors larger lambda.',
        interpretation='Development resampling stability, NOT a generalization confidence interval. Grid argmin frequencies are descriptive. This analysis NEVER changes the final selection, grid, endpoints, or evaluation.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes=dict(fixed_selection_file_sha256=hashlib.sha256(fixed_bytes).hexdigest(),
            development_provenance_sha256=fingerprint(data['provenance']),
            development_risk_arrays_sha256=fingerprint(dict(volume_ids=data['included'],
                clean=data['clean'].tolist(), combined=data['combined'].tolist())),
            configuration_sha256=data['config_hash']),
        input_hash_location='The fixed_selection.json provenance map contains the individual input file hashes; its exact file digest is recorded above.')
    if fixed_path.read_bytes() != fixed_bytes:
        raise ValueError('Fixed calibration changed during descriptive resampling')
    destination = root / 'calibration'
    with (destination / 'selection_stability.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['frequencies'][0]))
        writer.writeheader(); writer.writerows(result['frequencies'])
    atomic_json(destination / 'selection_stability.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    result = write_selection_stability(args.output)
    print(json.dumps(dict(status=result['status'], bootstrap_samples=result['bootstrap_samples'],
                         original_fixed_lambda=result['original_fixed_lambda'], frequencies=result['frequencies']), sort_keys=True))


if __name__ == '__main__':
    main()
