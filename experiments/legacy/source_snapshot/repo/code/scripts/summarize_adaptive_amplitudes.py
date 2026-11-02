"""Post-controller descriptive report for the two planned amplitude conditions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import audit_adaptive as audit
import summarize_adaptive as report
from adaptive_study import OLD
from calibrate_adaptive import ROOT, OUT, _read
from complete_adaptive_study import PIPELINE_SCRIPTS
from spine_registration.benchmark import atomic_json, fingerprint

AMPLITUDES = {'small_2deg': 'small', 'stress_16deg': 'large'}
ADDENDUM = 'AMPLITUDE_REPORTING_ADDENDUM.md'


def _original_target(root, case, scenario, manifest, base_hash, provenance, checked_sources, counts):
    """CPU-only verification of the original and copied arrays, including rejections."""
    directory = root / 'targets' / scenario / case
    original_directory = OLD / 'targets' / AMPLITUDES[scenario] / case
    metadata = _read(directory / 'metadata.json', provenance)
    original = _read(original_directory / 'metadata.json', provenance)
    if (metadata.get('case') != case or original.get('case') != case
            or metadata.get('scenario') != scenario or original.get('scenario') != AMPLITUDES[scenario]
            or metadata.get('base_scenario') != AMPLITUDES[scenario]
            or metadata.get('base_config_hash') != base_hash
            or metadata.get('adaptive_config_hash') != manifest['config_hash']
            or metadata.get('robustness_manifest_hash') != fingerprint(manifest)
            or (ROOT / metadata.get('source_target', '')).resolve() != original_directory.resolve()
            or metadata.get('original_amplitude_target_hash') != fingerprint(original)
            or type(original.get('accepted')) is not bool or metadata.get('accepted') != original['accepted']
            or fingerprint(metadata.get('motion')) != fingerprint(original.get('motion'))):
        raise ValueError('Amplitude original target identity/motion/provenance mismatch')
    for source, expected in [(directory, metadata), (original_directory, original)]:
        if audit._target(source, base_hash, checked_sources, counts) != expected:
            raise ValueError('Amplitude metadata changed during validation')
        if expected['accepted']:
            for view in ('AP', 'LAT'):
                path = source / f'{view}.npy'
                provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata['accepted'] and metadata['image_hashes'] != original['image_hashes']:
        raise ValueError('Amplitude copied image hashes differ from original target')
    return metadata


def summarize_amplitudes(output_root=OUT, *, bootstrap_samples=10000):
    root, provenance = Path(output_root), {}
    controller_path, summary_path = root / 'controller_status.json', root / 'summary/summary.json'
    for path in (controller_path, summary_path):
        if not path.is_file():
            raise ValueError('Require COMPLETE controller and matching COMPLETE main summary')
    controller = _read(controller_path, provenance)
    if controller.get('status') != 'COMPLETE' or controller.get('active_stage') is not None:
        raise ValueError('Require COMPLETE controller before amplitude reporting')
    summary = _read(summary_path, provenance)
    if summary.get('status') != 'COMPLETE':
        raise ValueError('Require matching COMPLETE main summary')
    config, config_hash, split, selection, _, methods = report._identity(root, provenance)
    expected_sources = {name: hashlib.sha256((ROOT / 'code/scripts' / name).read_bytes()).hexdigest()
                        for name in PIPELINE_SCRIPTS}
    stages = ['development', 'calibrate', 'evaluate', 'sensitivity', 'robustness_prepare',
              'robustness_run', 'robustness_scales', 'summarize']
    if selection.get('adaptive') is not None:
        stages += ['adaptive_oof', 'adaptive_select']
    if (controller.get('config_hash') != config_hash or controller.get('source_hashes') != expected_sources
            or any(controller.get('stages', {}).get(stage, {}).get('status') != 'COMPLETE' for stage in stages)):
        raise ValueError('Controller source/configuration/stage completion mismatch')
    if (summary.get('config_hash') != config_hash or summary.get('selection') != selection
            or summary.get('methods') != methods
            or summary.get('reporter_sha256') != expected_sources['summarize_adaptive.py']):
        raise ValueError('Main summary identity/selection/reporter mismatch')
    manifest_path = root / 'robustness/manifest.json'
    manifest = _read(manifest_path, provenance)
    names = split['evaluation'][:10]
    if (manifest.get('config_hash') != config_hash or manifest.get('base_config_hash') != config['base_config_hash']
            or manifest.get('cases') != names or len(names) != 10
            or manifest.get('amplitude_scenarios') != AMPLITUDES
            or manifest.get('selected_lambda') != selection['fixed_lambda']
            or manifest.get('selection_sha256') != provenance[str(root / 'calibration/selection.json')]
            or manifest.get('split_sha256') != provenance[str(root / 'split.json')]):
        raise ValueError('Amplitude manifest/configuration/selection/first-ten coverage mismatch')
    if not manifest.get('source_hashes'):
        raise ValueError('Missing amplitude implementation provenance')
    for relative, digest in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Amplitude source changed since manifest freeze')
    if any(summary.get('provenance', {}).get(path) != digest for path, digest in provenance.items()
           if path not in (str(controller_path), str(summary_path))):
        raise ValueError('Main summary provenance differs from amplitude prerequisites')
    datasets, rows, volumes, thresholds, checked_sources = {}, [], [], [], set()
    counts = dict(target_arrays_checked=0)
    for scenario in AMPLITUDES:
        selected, included, excluded = [], [], []
        for case in names:
            metadata = _original_target(root, case, scenario, manifest, config['base_config_hash'],
                                        provenance, checked_sources, counts)
            eligibility, target_rows, _ = report._target_records(root, case, scenario, config, config_hash,
                                                                selection, methods, provenance, manifest)
            if Path(eligibility['source']).resolve() != (root / 'targets' / scenario / case).resolve():
                raise ValueError('Amplitude eligibility references a different prepared target')
            if eligibility['accepted'] != metadata['accepted']:
                raise ValueError('Amplitude eligibility differs from original target')
            if not eligibility['accepted']:
                excluded.append(case)
                continue
            if len(target_rows) != len(methods) or {row['method'] for row in target_rows} != set(methods):
                raise ValueError('Incomplete paired amplitude method coverage')
            included.append(case)
            selected.extend(target_rows)
        dataset = dict(status='COMPLETE_DESCRIPTIVE' if selected else 'NO_ELIGIBLE_TARGETS',
            n_assigned=10, n_eligible=len(included), included_cases=included, excluded_cases=excluded,
            geometry_coverage=len(included) / 10, inference='descriptive; no additional confirmatory tests')
        if selected:
            described, local_volumes, local_thresholds = report._describe(selected, 10, scenario, bootstrap_samples)
            dataset['methods'] = described
            rows.extend(dict(dataset=scenario, **row) for row in selected)
            volumes.extend(local_volumes)
            thresholds.extend(local_thresholds)
        else:
            dataset.update(methods={}, pipeline_success_frequency_all_assigned=0)
        datasets[scenario] = dataset
    addendum_path = root / ADDENDUM
    provenance[str(addendum_path)] = hashlib.sha256(addendum_path.read_bytes()).hexdigest()
    result = dict(status='COMPLETE_DESCRIPTIVE_ONLY', config_hash=config_hash, methods=methods,
        assigned_cases=names, assigned_targets=20, verified_targets=sum(x['n_assigned'] for x in datasets.values()),
        scenarios=datasets, bootstrap_samples=bootstrap_samples, provenance=provenance,
        source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), Path(audit.__file__), Path(report.__file__))},
        original_and_copied_target_arrays_checked=counts['target_arrays_checked'],
        interpretation='Planned 2/16-degree sensitivity omitted from the frozen report loop. Separate post-controller descriptive report; all 20 assigned targets retained, exclusions counted as pipeline failures without invented RMS. No reselection or new inference.',
        addendum=dict(path=str(addendum_path), sha256=provenance[str(addendum_path)]))
    for path in (controller_path, summary_path, manifest_path):
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance[str(path)]:
            raise ValueError('Amplitude reporting prerequisites changed during validation')
    destination = root / 'summary'
    report._csv(destination / 'amplitude_rows.csv', rows)
    report._csv(destination / 'amplitude_per_volume.csv', volumes)
    report._csv(destination / 'amplitude_thresholds.csv', thresholds)
    atomic_json(destination / 'amplitude_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    result = summarize_amplitudes(args.output)
    print(json.dumps(dict(status=result['status'], verified_targets=result['verified_targets']), sort_keys=True))


if __name__ == '__main__':
    main()
