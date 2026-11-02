"""Complete the predeclared descriptive lambda=.01 comparison after the main study."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time


OUT = Path(__file__).resolve().parents[2] / 'outputs/adaptive_prior_study'
SCENARIOS = ['clean'] + [f'combined_s{i}' for i in range(5)]
SCOPE = 'descriptive_only_no_hypothesis_test'


def _read(path):
    if not path.is_file():
        raise ValueError(f'Missing prerequisite: {path}')
    return json.loads(path.read_text())


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _lock(path, *, shared=False):
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another study process holds {path.name}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _context(root):
    # These checks deliberately precede heavy imports and any evaluation-record reads.
    controller_path = root / 'controller_status.json'
    controller = _read(controller_path)
    if controller.get('status') != 'COMPLETE':
        raise ValueError('The main controller must be COMPLETE before lambda=.01 execution/reporting')
    config = _read(root / 'config.json')
    config_hash = _fingerprint(config)
    if controller.get('config_hash') != config_hash:
        raise ValueError('Controller/configuration hash mismatch')
    main_path = root / 'summary/summary.json'
    main = _read(main_path)
    if main.get('status') != 'COMPLETE' or main.get('config_hash') != config_hash:
        raise ValueError('A COMPLETE main summary with matching configuration is required')
    cases = config['split']['evaluation']
    if (len(cases) != 50 or len(set(cases)) != 50
            or config['main_scenarios'] != SCENARIOS or .01 not in config['lambda_grid']):
        raise ValueError('Expected the frozen 50 evaluation IDs, six main scenarios, and lambda=.01 grid entry')
    identity = dict(config_hash=config_hash, controller_sha256=_sha256(controller_path),
                    main_summary_sha256=_sha256(main_path), helper_sha256=_sha256(Path(__file__)))
    return config, config_hash, main, identity


def _receipt_fields(config, identity, shard):
    return dict(identity=identity, shard=shard, shards=4,
                cases=config['split']['evaluation'][shard::4], scenarios=SCENARIOS, **{'lambda': .01})


def _check_receipt(receipt, expected, *, complete=False):
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Lambda=.01 shard receipt has stale identity, partition, or configuration')
    if complete and receipt.get('status') != 'COMPLETE':
        raise ValueError('All four lambda=.01 shards must be COMPLETE before reporting')


def run(output_root=OUT, *, shard=0, shards=4):
    if shards != 4 or type(shard) is not int or not 0 <= shard < 4:
        raise ValueError('Lambda=.01 execution requires exactly four shards, numbered 0 through 3')
    root = Path(output_root)
    with _lock(root / 'controller.lock', shared=True), _lock(root / f'lambda001_{shard}.lock'):
        config, config_hash, _, identity = _context(root)
        expected = _receipt_fields(config, identity, shard)
        path = root / f'lambda001_{shard}_completion.json'
        receipt = _read(path) if path.exists() else dict(**expected, attempts=[])
        _check_receipt(receipt, expected)

        import adaptive_study as study

        if study.OUT.resolve() != root.resolve():
            raise ValueError('Execution output must match the existing adaptive_study output root')
        initialized, initialized_hash, base, base_hash = study.initialize()
        if initialized != config or initialized_hash != config_hash:
            raise ValueError('Initialization differs from the completed main study')
        study.torch.set_num_threads(1)
        started = time.monotonic()
        attempt = dict(started_at=_now(), status='RUNNING', completed_cases=[])
        receipt['attempts'].append(attempt)
        receipt['status'] = 'RUNNING'
        _write(path, receipt)
        try:
            for case in expected['cases']:
                # Existing record() validates cached signatures before reuse on every resume.
                study.run(case, SCENARIOS, [.01], config, config_hash, base, base_hash, adaptive=None)
                attempt['completed_cases'].append(case)
                _write(path, receipt)
            if _context(root)[3] != identity:
                raise ValueError('Completed-main identity changed during lambda=.01 execution')
        except BaseException as error:
            attempt.update(status='FAILED', ended_at=_now(), elapsed_seconds=time.monotonic() - started,
                           error=f'{type(error).__name__}: {error}')
            receipt['status'] = 'FAILED'
            _write(path, receipt)
            raise
        attempt.update(status='COMPLETE', ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        receipt['status'] = 'COMPLETE'
        _write(path, receipt)
        return receipt


def _descriptive_comparisons(volumes):
    comparisons = []
    for dataset in ('clean', 'combined'):
        first = {row['case']: row for row in volumes
                 if row['dataset'] == dataset and row['method'] == 'C_lambda0.01'}
        for comparator in ('B', 'C_fixed', 'C_lambda0.1'):
            second = {row['case']: row for row in volumes
                      if row['dataset'] == dataset and row['method'] == comparator}
            if set(first) != set(second):
                raise ValueError('Descriptive comparison requires the same eligible volumes')
            differences = [dict(case=case,
                rms_difference_mm=first[case]['surface_rms_tre_mm'] - second[case]['surface_rms_tre_mm'],
                failure_frequency_difference=first[case]['failure_gt3mm'] - second[case]['failure_gt3mm'])
                for case in sorted(first)]
            comparisons.append(dict(dataset=dataset, first='C_lambda0.01', second=comparator,
                scope=SCOPE, n_volumes=len(differences), paired_differences=differences,
                mean_rms_difference_mm=sum(x['rms_difference_mm'] for x in differences) / len(differences),
                mean_failure_frequency_difference=sum(x['failure_frequency_difference'] for x in differences) / len(differences)))
    return comparisons


def summarize(output_root=OUT, *, bootstrap_samples=10000):
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError('At least 100 bootstrap samples are required')
    root, provenance = Path(output_root), {}
    with ExitStack() as stack:
        stack.enter_context(_lock(root / 'controller.lock', shared=True))
        stack.enter_context(_lock(root / 'lambda001_summary.lock'))
        config, config_hash, main, identity = _context(root)
        receipts = []
        for shard in range(4):
            stack.enter_context(_lock(root / f'lambda001_{shard}.lock', shared=True))
            receipt_path = root / f'lambda001_{shard}_completion.json'
            receipt = _read(receipt_path)
            _check_receipt(receipt, _receipt_fields(config, identity, shard), complete=True)
            receipts.append(receipt)
            provenance[str(receipt_path)] = _sha256(receipt_path)

        import summarize_adaptive as reporting

        frozen, frozen_hash, split, selection, _, methods = reporting._identity(root, provenance)
        if frozen != config or frozen_hash != config_hash:
            raise ValueError('Frozen reporting identity differs from the completed main study')
        methods = {**methods, 'C_lambda0.01': 'lambda_0.01'}
        rows, included, excluded = [], [], []
        for case in split['evaluation']:
            statuses, case_rows = [], []
            for scenario in SCENARIOS:
                status, target_rows, _ = reporting._target_records(
                    root, case, scenario, config, config_hash, selection, methods, provenance)
                statuses.append(status)
                case_rows.extend(target_rows)
            if len({status['accepted'] for status in statuses}) != 1:
                raise ValueError('Evaluation geometry eligibility differs across the six main targets')
            if not statuses[0]['accepted']:
                if len({_fingerprint(status['motion']) for status in statuses}) != 1:
                    raise ValueError('Evaluation exclusion evidence differs across targets')
                excluded.append(case)
            else:
                included.append(case)
                rows.extend(case_rows)
        coverage = main['coverage']
        if (len(included) < 2 or coverage['n_assigned_evaluation'] != 50
                or included != coverage['included_cases'] or excluded != coverage['excluded_cases']):
            raise ValueError('Lambda=.01 eligibility must match the COMPLETE main summary')
        descriptions, volumes, thresholds = {}, [], []
        for dataset in ('clean', 'combined'):
            selected = [row for row in rows if (row['scenario'] == 'clean') == (dataset == 'clean')]
            description, volume_rows, threshold_rows = reporting._describe(selected, 50, dataset, bootstrap_samples)
            descriptions[dataset] = description
            volumes.extend(volume_rows)
            thresholds.extend(threshold_rows)
        if _context(root)[3] != identity:
            raise ValueError('Completed-main identity changed during lambda=.01 reporting')
        result = dict(status='COMPLETE_DESCRIPTIVE', scope=SCOPE, generated_at=_now(),
            config_hash=config_hash, identity=identity, methods=methods, coverage=coverage,
            interpretation='Predeclared fixed lambda=.01 sensitivity; no new hypothesis test, gate, endpoint, or parameter selection. Negative paired differences favor lambda=.01.',
            bootstrap_samples=bootstrap_samples, main=descriptions, failure_thresholds=thresholds,
            comparisons=_descriptive_comparisons(volumes), shard_receipts=receipts,
            provenance=provenance, reporter_sha256=_sha256(Path(reporting.__file__)))
        destination = root / 'summary'
        for name, data in (('rows', rows), ('per_volume', volumes)):
            path = destination / f'lambda001_{name}.csv'
            temporary = path.with_suffix('.csv.tmp')
            try:
                reporting._csv(temporary, data)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        _write(destination / 'lambda001_summary.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    execute = subparsers.add_parser('run', help='Run one of exactly four resumable shards after COMPLETE')
    execute.add_argument('--shard', type=int, required=True)
    execute.add_argument('--shards', type=int, default=4)
    execute.add_argument('--output', type=Path, default=OUT)
    report = subparsers.add_parser('summarize', help='Write separate descriptive summaries after all shards complete')
    report.add_argument('--output', type=Path, default=OUT)
    report.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.output, shard=args.shard, shards=args.shards)
    else:
        summarize(args.output, bootstrap_samples=args.bootstrap_samples)


if __name__ == '__main__':
    main()
