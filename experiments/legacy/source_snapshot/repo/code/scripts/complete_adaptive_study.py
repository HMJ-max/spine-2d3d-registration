"""Resume existing development workers, then complete the locked study in order."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code/scripts"
OUT = ROOT / "outputs/adaptive_prior_study"
PIPELINE_SCRIPTS = ("complete_adaptive_study.py", "adaptive_study.py", "calibrate_adaptive.py",
                    "adaptive_oof.py", "adaptive_robustness.py", "summarize_adaptive.py")


def _read(path):
    return json.loads(Path(path).read_text())


def _save(output, state):
    with tempfile.NamedTemporaryFile(mode="w", dir=output, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "controller_status.json")


def _running(process):
    directory = Path("/proc") / str(process["pid"])
    try:
        command = (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        status = (directory / "stat").read_text().rsplit(") ", 1)[1][0]
        return status not in ("Z", "X") and command == process["command"]
    except FileNotFoundError:
        return False


def _wait_development(output, config, config_hash, poll_seconds):
    processes = _read(output / "development_processes.json")
    if sorted(row["shard"] for row in processes) != list(range(4)):
        raise ValueError("Require exactly four recorded development workers")
    while True:
        pending = []
        for process in processes:
            shard = process["shard"]
            path = output / f"develop_{shard}_completion.json"
            if path.exists():
                row = _read(path)
                if (row.get("status") != "COMPLETE" or row.get("config_hash") != config_hash
                        or row.get("cases") != config["split"]["development"][shard::4]
                        or row.get("scenarios") != config["main_scenarios"]):
                    raise ValueError(f"Invalid development completion: {path}")
            else:
                if not _running(process) and not path.exists():
                    raise RuntimeError(f"Development shard {shard} disappeared or exited without valid completion; pid={process['pid']}")
                pending.append(shard)
        if not pending:
            return
        print(f"Waiting for existing development shards: {pending}", flush=True)
        time.sleep(poll_seconds)


def _stage(name, commands, state, output, poll_seconds):
    previous = state["stages"].get(name, {})
    if previous.get("status") == "COMPLETE":
        return
    if any(_running(process) for process in previous.get("processes", [])):
        raise RuntimeError(f"Prior {name} workers are still running; refusing duplicate launch")
    stage = state["stages"][name] = dict(status="RUNNING", processes=[])
    state["active_stage"] = name
    _save(output, state)
    logs = output / "controller_logs"
    logs.mkdir(exist_ok=True)
    children = []
    try:
        for shard, command in enumerate(commands):
            log = logs / f"{name}_{shard}.log"
            with log.open("ab") as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            children.append(child)
            stage["processes"].append(dict(pid=child.pid, command=command, log=str(log)))
            _save(output, state)
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"{name} failed with exit codes {codes}; see {logs}/{name}_*.log")
            if all(code == 0 for code in codes):
                stage.update(status="COMPLETE", exit_codes=codes)
                _save(output, state)
                print(f"Completed {name}", flush=True)
                return
            print(f"Running {name}: {sum(code is None for code in codes)} workers", flush=True)
            time.sleep(poll_seconds)
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        stage.update(status="FAILED", exit_codes=[child.returncode for child in children])
        _save(output, state)
        raise


def complete(output=OUT, *, poll_seconds=30.):
    output = Path(output)
    if not 0 < poll_seconds <= 60:
        raise ValueError("Polling interval must be greater than zero and at most 60 seconds")
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = _read(output / "config.json")
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        sources = {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in PIPELINE_SCRIPTS}
        path = output / "controller_status.json"
        state = _read(path) if path.exists() else dict(config_hash=config_hash, source_hashes=sources, stages={})
        if state.get("config_hash") != config_hash or state.get("source_hashes") != sources:
            raise ValueError("Controller configuration or pipeline source changed; refusing stale progress")
        state.update(status="RUNNING", active_stage="development")
        state.pop("error", None)
        _save(output, state)

        def execute(name, script, *arguments, shards=False):
            if hashlib.sha256((SCRIPTS / script).read_bytes()).hexdigest() != sources[script]:
                raise ValueError(f"Pipeline source changed during execution: {script}")
            commands = [[sys.executable, "-u", str(SCRIPTS / script), *arguments,
                         *(["--shards", "4", "--shard", str(shard)] if shards else [])]
                        for shard in range(4 if shards else 1)]
            _stage(name, commands, state, output, poll_seconds)

        try:
            _wait_development(output, config, config_hash, poll_seconds)
            state["stages"]["development"] = dict(status="COMPLETE")
            _save(output, state)
            execute("calibrate", "calibrate_adaptive.py", "--output", str(output))
            gate = _read(output / "calibration/proxy_gate.json")
            if gate.get("config_hash") != config_hash or type(gate.get("gate_passed")) is not bool:
                raise ValueError("Missing or inconsistent pooled proxy gate")
            if gate["gate_passed"]:
                execute("adaptive_oof", "adaptive_oof.py", "run", "--output", str(output), shards=True)
                execute("adaptive_select", "adaptive_oof.py", "select", "--output", str(output))
            selection = _read(output / "calibration/selection.json")
            if selection.get("config_hash") != config_hash or selection.get("proxy_gate_passed") is not gate["gate_passed"]:
                raise ValueError("Final calibration selection is missing or inconsistent")
            for action in ("evaluate", "sensitivity"):
                execute(action, "adaptive_study.py", action, shards=True)
            for action in ("prepare", "run", "scales"):
                execute(f"robustness_{action}", "adaptive_robustness.py", action, shards=True)
            execute("summarize", "summarize_adaptive.py", "--output", str(output))
            state.update(status="COMPLETE", active_stage=None)
        except BaseException as error:
            state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            _save(output, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    complete(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
