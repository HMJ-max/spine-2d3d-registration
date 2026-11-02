"""Finish computations and validation independently of the assistant session."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/adaptive_prior_study'


def ready(value):
    if value['status'] == 'FAILED':
        raise RuntimeError(str(value.get('error', 'Main controller failed')))
    if value['status'] not in ('RUNNING', 'COMPLETE'):
        raise ValueError('Unexpected controller status')
    return value['status'] == 'COMPLETE'


def read(path):
    return json.loads(path.read_text())


def save(state):
    path = OUT / 'finish_reporting_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    with (OUT / 'finish_reporting.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='RUNNING', active_stage='wait_main_controller', pid=os.getpid(), stages={})
        save(state)
        logs = OUT / 'finish_reporting_logs'
        logs.mkdir(exist_ok=True)
        try:
            while not ready(read(OUT / 'controller_status.json')):
                os.kill(read(OUT / 'controller_process.json')['pid'], 0)
                time.sleep(30)
            stages = [
                ('lambda001', 'adaptive_lambda001.py', [['run', '--shards', '4', '--shard', str(i)] for i in range(4)]),
                ('lambda001_summary', 'adaptive_lambda001.py', [['summarize']]),
                ('amplitude_summary', 'summarize_adaptive_amplitudes.py', [[]]),
                ('independent_audit', 'audit_adaptive.py', [[]]),
                ('replay', 'replay_adaptive.py', [[]]),
                ('proxy_diagnostics', 'adaptive_proxy_diagnostics.py', [[]]),
            ]
            for name, script, arguments in stages:
                state['active_stage'] = name
                processes, handles = [], []
                try:
                    for index, args in enumerate(arguments):
                        command = [sys.executable, '-u', str(ROOT / 'code/scripts' / script), *args]
                        handle = (logs / f'{name}_{index}.log').open('a')
                        handles.append(handle)
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                   stdout=handle, stderr=subprocess.STDOUT)
                        processes.append(process)
                    state['stages'][name] = dict(status='RUNNING', pids=[p.pid for p in processes])
                    save(state)
                    while any(p.poll() is None for p in processes):
                        time.sleep(15)
                    codes = [p.returncode for p in processes]
                    state['stages'][name].update(status='COMPLETE' if all(c == 0 for c in codes) else 'FAILED', exit_codes=codes)
                    save(state)
                    if any(c != 0 for c in codes):
                        raise RuntimeError(f'{name} failed with exit codes {codes}')
                finally:
                    for handle in handles:
                        handle.close()
            state.update(status='COMPLETE', active_stage=None)
        except BaseException as error:
            state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save(state)


if __name__ == '__main__':
    main()
