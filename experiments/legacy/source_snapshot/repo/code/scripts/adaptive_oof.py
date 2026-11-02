"""Gated development-only out-of-fold adaptive-prior calibration."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_analysis import LAMBDA_GRID, select_lambda, training_tau
from calibrate_adaptive import OUT, ROOT, SCENARIOS, _read, _record, _write_locked, load_development
from spine_registration.benchmark import fingerprint


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _method(weight):
    return f"adaptive_oof_{weight:g}"


def _locked_development(root):
    provenance = {}
    gate = _read(root / "calibration/proxy_gate.json", provenance)
    if gate.get("gate_passed") is not True:
        raise ValueError("The pooled proxy gate must pass before adaptive OOF calibration")
    fixed = _read(root / "calibration/fixed_selection.json", provenance)
    if fixed.get("status") != "FIXED_CALIBRATION_COMPLETE":
        raise ValueError("Fixed development calibration must be complete")
    data = load_development(root)
    for artifact in (fixed, gate):
        if (artifact.get("config_hash") != data["config_hash"]
                or artifact.get("included_cases") != data["included"]
                or artifact.get("excluded_cases") != data["excluded"]):
            raise ValueError("Calibration configuration or eligibility changed")
    if (fixed.get("provenance") != data["provenance"]
            or fixed.get("n_assigned_development") != len(data["config"]["split"]["development"])
            or fixed.get("selection") != select_lambda(data["clean"], data["combined"])
            or fixed.get("fixed_lambda") != fixed["selection"]["selected_lambda"]):
        raise ValueError("Fixed calibration is incomplete or development inputs changed")
    return data, fixed, gate


def _fold_plan(ids, folds, proxies):
    """Fit each tau from every edge/target of eligible training volumes only."""
    planned = []
    for index, validation in enumerate(folds):
        training = [case for case in ids if case not in validation]
        held = [case for case in ids if case in validation]
        if not held or len(training) < 2:
            raise ValueError("Every fold needs eligible validation and at least two training volumes")
        values = np.asarray([proxies[case] for case in training], float)
        planned.append(dict(fold=index, training_cases=training, validation_cases=held,
                            tau=training_tau(values), training_edge_observations=int(values.size)))
    if sorted(case for fold in planned for case in fold["validation_cases"]) != sorted(ids):
        raise ValueError("Every eligible development volume must be held out exactly once")
    return planned


def prepare(output_root=OUT):
    """Validate completed development and lock the complete plan before refinement."""
    root = Path(output_root)
    # Read-only gate checks precede even creating a lock or inspecting outcomes.
    data, fixed, gate = _locked_development(root)
    source_hashes = {str(Path(__file__).resolve().relative_to(ROOT)): _sha256(__file__)}
    selections, proxies, checked = {}, {}, {}
    for case in data["included"]:
        proxies[case] = []
        for scenario in SCENARIOS:
            path = root / "banks" / scenario / case / "selection.json"
            selection = _read(path, checked)
            if checked[str(path)] != data["provenance"].get(str(path)):
                raise ValueError(f"Candidate selection changed during preparation: {path}")
            selections[case, scenario] = selection
            proxies[case].append(selection["proxy_edges"])
    folds = _fold_plan(data["included"], data["config"]["split"]["development_folds"], proxies)
    fold_for = {case: fold for fold in folds for case in fold["validation_cases"]}
    targets = []
    for case in data["included"]:
        fold = fold_for[case]
        for scenario in SCENARIOS:
            selection = selections[case, scenario]
            edges = np.asarray(selection["proxy_edges"], float)
            winner = root / "banks" / scenario / case / f"start_{selection['best']}.json"
            targets.append(dict(case=case, scenario=scenario, fold=fold["fold"], tau=fold["tau"],
                target_hash=selection["target_hash"], bank_winner=str(winner),
                bank_winner_sha256=data["provenance"][str(winner)], proxy_edges=edges.tolist(),
                edge_weights={_method(weight): (weight * edges / (edges + fold["tau"])).tolist()
                              for weight in LAMBDA_GRID[1:]}))
    manifest = dict(version="adaptive_oof_v1", config_hash=data["config_hash"],
        fixed_lambda=fixed["fixed_lambda"], fixed_selection_hash=fingerprint(fixed),
        proxy_gate_hash=fingerprint(gate), source_hashes=source_hashes,
        frozen_source_hashes=data["config"]["source_hashes"], lambda_grid=list(LAMBDA_GRID),
        scenarios=list(SCENARIOS), optimizer=data["config"]["optimizer"], levels=data["config"]["levels"],
        included_cases=data["included"], excluded_cases=data["excluded"], folds=folds, targets=targets,
        zero_lambda="reuse strong B", development_provenance=data["provenance"])
    with (root / "calibration/adaptive_oof.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_locked(root / "calibration/adaptive_oof_manifest.json", manifest)
    return data, manifest, proxies


def run(output_root=OUT, *, shard=0, shards=1):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    root = Path(output_root)
    data, manifest, _ = prepare(root)
    # Import the GPU runner only after gate, source and manifest validation.
    import adaptive_study as runner

    runner.torch.set_num_threads(1)
    base, base_hash = runner.frozen_config()
    if (base_hash != data["config"]["base_config_hash"] or base["BC"] != manifest["optimizer"]
            or base["levels"] != manifest["levels"]):
        raise ValueError("Frozen optimizer, levels or base configuration changed")
    names = data["included"][shard::shards]
    for name in names:
        with (root / "calibration" / f"adaptive_oof_{name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            case = runner.verified_case(name)
            problem = runner.ImageProblem(case, base)
            for target in (row for row in manifest["targets"] if row["case"] == name):
                scenario, checked = target["scenario"], {}
                eligibility_path = root / "eligibility" / scenario / f"{name}.json"
                eligible = _read(eligibility_path, checked)
                winner = _read(Path(target["bank_winner"]), checked)
                if any(digest != data["provenance"].get(path) for path, digest in checked.items()):
                    raise ValueError("Locked eligibility or bank winner changed")
                images, meta = runner.checked_target(Path(eligible["source"]), base_hash)
                if images is None or fingerprint(meta) != target["target_hash"]:
                    raise ValueError("Locked development target changed")
                case["gt_params"] = np.asarray(meta["motion"]["params"])
                case["gt_matrices"] = np.asarray(meta["motion"]["transforms"])
                problem.set_targets(images)
                initial = np.asarray(winner["pose"])
                for method, weights in target["edge_weights"].items():
                    runner.record(root / "results" / scenario / name / f"{method}.json", method,
                                  problem, case, meta, data["config_hash"], initial=initial, weights=weights)
    return dict(status="OOF_SHARD_COMPLETE", shard=shard, shards=shards, cases=names,
                manifest_hash=fingerprint(manifest))


def select(output_root=OUT):
    root = Path(output_root)
    if not (root / "calibration/adaptive_oof_manifest.json").is_file():
        raise ValueError("Missing pre-execution adaptive OOF manifest")
    data, manifest, proxies = prepare(root)
    clean, combined = data["clean"].copy(), data["combined"].copy()
    provenance, checked = {}, {}
    source_hash = data["config"]["source_hashes"]["code/scripts/adaptive_study.py"]
    case_index = {case: index for index, case in enumerate(data["included"])}
    for target in manifest["targets"]:
        case, scenario = target["case"], target["scenario"]
        winner = _read(Path(target["bank_winner"]), checked)
        if checked[str(target["bank_winner"])] != target["bank_winner_sha256"]:
            raise ValueError("Locked bank winner changed")
        for column, weight in enumerate(LAMBDA_GRID[1:], 1):
            method = _method(weight)
            path = root / "results" / scenario / case / f"{method}.json"
            row = _record(path, method, case, scenario, target["target_hash"], data["config_hash"],
                          source_hash, provenance)
            weights = target["edge_weights"][method]
            signature = fingerprint(dict(method=method, case=case, weights=weights, scales=[3., 3.],
                initial=np.asarray(winner["pose"]).tolist(), imported_sha256=None, implementation_sha256=source_hash))
            if (row.get("edge_weights") != weights or row.get("scales") != [3., 3.]
                    or row["input_signature"] != signature
                    or [level["level"] for level in row["optimization"]["levels"]] != manifest["levels"]):
                raise ValueError(f"OOF weights, initialization or levels differ from the locked plan: {path}")
            risk = row["metrics"]["surface_rms_tre_mm"]
            index = case_index[case]
            if scenario == "clean":
                clean[index, column] = risk
            else:
                combined[index, column, SCENARIOS.index(scenario) - 1] = risk
    selection = select_lambda(clean, combined)
    tau = training_tau([proxies[case] for case in data["included"]])
    final = dict(status="READY_FOR_ADAPTIVE_EVALUATION", config_hash=data["config_hash"],
        fixed_lambda=manifest["fixed_lambda"], fixed_selection_hash=manifest["fixed_selection_hash"],
        proxy_gate_hash=manifest["proxy_gate_hash"], proxy_gate_passed=True,
        adaptive=dict(lambda_max=selection["selected_lambda"], tau=tau),
        included_cases=data["included"], excluded_cases=data["excluded"],
        selection=selection, oof_provenance=dict(manifest_hash=fingerprint(manifest),
            manifest_path=str(root / "calibration/adaptive_oof_manifest.json"),
            source_hashes=manifest["source_hashes"], folds=manifest["folds"], result_hashes=provenance,
            zero_lambda="reuse strong B", final_tau_edge_observations=len(data["included"]) * len(SCENARIOS) * 4))
    with (root / "calibration/adaptive_oof.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_locked(root / "calibration/selection.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "select"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    result = run(args.output, shard=args.shard, shards=args.shards) if args.action == "run" else select(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
