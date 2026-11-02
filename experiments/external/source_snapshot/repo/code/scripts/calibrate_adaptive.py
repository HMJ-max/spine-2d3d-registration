"""Strict development-only fixed-prior calibration and proxy feasibility gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from spine_registration.benchmark import atomic_json, fingerprint
from adaptive_analysis import LAMBDA_GRID, proxy_gate, select_lambda

OUT = ROOT / "outputs/adaptive_prior_study"
SCENARIOS = ("clean", "combined_s0", "combined_s1", "combined_s2", "combined_s3", "combined_s4")


def _read(path, provenance):
    if not path.is_file():
        raise ValueError(f"Incomplete development experiment: missing {path}")
    content = path.read_bytes()
    provenance[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def _record(path, method, case, scenario, target_hash, config_hash, source_hash, provenance):
    row = _read(path, provenance)
    expected = dict(case=case, scenario=scenario, method=method,
                    target_hash=target_hash, config_hash=config_hash, implementation_sha256=source_hash)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Record identity/config/target mismatch: {path}")
    if not isinstance(row.get("input_signature"), str) or len(row["input_signature"]) != 64:
        raise ValueError(f"Missing checkpoint input signature: {path}")
    pose = np.asarray(row["pose"], float)
    if pose.shape != ((6,) if method == "A" else (5, 6)) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose: {path}")
    vertices = row["vertebrae"]
    if [v["label"] for v in vertices] != list(range(20, 25)):
        raise ValueError(f"Require exactly five ordered vertebrae: {path}")
    rms = np.asarray([v["surface_rms_tre_mm"] for v in vertices], float)
    errors = np.asarray([v["translation_error_mm"] for v in vertices], float)
    if not np.isfinite(rms).all() or not np.isfinite(errors).all() or np.any(rms < 0) or np.any(errors < 0):
        raise ValueError(f"Invalid vertebral metrics: {path}")
    metrics = row["metrics"]
    if (not np.isclose(metrics["surface_rms_tre_mm"], rms.mean(), rtol=1e-8, atol=1e-10)
            or metrics["case_failure_gt3mm"] not in (False, True)
            or bool(metrics["case_failure_gt3mm"]) != bool(np.any(errors > 3))):
        raise ValueError(f"Metric aggregation/failure definition mismatch: {path}")
    if not np.isfinite(row["ncc_loss"]):
        raise ValueError(f"Nonfinite candidate loss: {path}")
    return row


def load_development(output_root=OUT):
    """Load all DEV rows, rejecting missing outcomes and inconsistent exclusions.

    Evaluation result directories are never enumerated or opened. Eligibility
    must be explicit for every development volume in all six main scenarios.
    """
    root, provenance = Path(output_root), {}
    config = _read(root / "config.json", provenance)
    config_hash = fingerprint(config)
    split = _read(root / "split.json", provenance)
    if split != config["split"] or tuple(config["lambda_grid"]) != LAMBDA_GRID or tuple(config["main_scenarios"]) != SCENARIOS:
        raise ValueError("Split, lambda grid or scenarios differ from the locked configuration")
    dev, evaluation, folds = split["development"], split["evaluation"], split["development_folds"]
    if (len(dev) != 25 or len(set(dev)) != 25 or len(evaluation) != 50 or len(set(evaluation)) != 50
            or set(dev) & set(evaluation) or len(folds) != 5 or sorted(sum(folds, [])) != sorted(dev)):
        raise ValueError("Require disjoint 25/50 split and five complete development folds")
    for relative, expected_hash in config["source_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen source changed: {relative}")
    source_hash = config["source_hashes"]["code/scripts/adaptive_study.py"]
    clean, combined, included, excluded, scores, failures, score_ids = [], [], [], [], [], [], []
    for case in dev:
        eligibility = [_read(root / "eligibility" / scenario / f"{case}.json", provenance) for scenario in SCENARIOS]
        if any(row.get("case") != case or type(row.get("accepted")) is not bool for row in eligibility):
            raise ValueError(f"Invalid eligibility identity/status for {case}")
        accepted = [row["accepted"] for row in eligibility]
        if len(set(accepted)) != 1:
            raise ValueError(f"Geometry eligibility must be common to all six targets: {case}")
        if not accepted[0]:
            if any(not isinstance(row.get("motion"), dict) or row["motion"].get("accepted") is not False for row in eligibility):
                raise ValueError(f"Missing geometry rejection evidence for {case}")
            if len({fingerprint(row["motion"]) for row in eligibility}) != 1:
                raise ValueError(f"Geometry rejection evidence differs across targets: {case}")
            excluded.append(case)
            continue
        risks = []
        for scenario, eligible in zip(SCENARIOS, eligibility):
            target_hash = eligible.get("target_hash")
            if not isinstance(target_hash, str) or len(target_hash) != 64:
                raise ValueError(f"Missing target fingerprint for {case}/{scenario}")
            target_directory = Path(eligible["source"])
            if target_directory.name != case:
                raise ValueError(f"Target metadata must belong to the development case: {case}")
            metadata = _read(target_directory / "metadata.json", provenance)
            allowed_scenario = {"clean": "moderate", "combined_s0": "combined"}.get(scenario, scenario)
            if (metadata.get("case") != case or metadata.get("scenario") != allowed_scenario
                    or metadata.get("accepted") is not True or metadata.get("config_hash") != config["base_config_hash"]
                    or fingerprint(metadata) != target_hash):
                raise ValueError(f"Target metadata/config fingerprint mismatch: {case}/{scenario}")
            directory = root / "results" / scenario / case
            bank = root / "banks" / scenario / case
            records = {}
            for method in ["A", "Zero", "B"] + [f"lambda_{weight:g}" for weight in LAMBDA_GRID[1:]]:
                records[method] = _record(directory / f"{method}.json", method, case, scenario, target_hash,
                                          config_hash, source_hash, provenance)
            candidates = [_record(bank / f"start_{k}.json", f"start_{k}", case, scenario, target_hash,
                                   config_hash, source_hash, provenance) for k in range(5)]
            selection = _read(bank / "selection.json", provenance)
            if any(selection.get(key) != value for key, value in dict(case=case, scenario=scenario,
                    config_hash=config_hash, target_hash=target_hash).items()):
                raise ValueError(f"Candidate selection identity/config/target mismatch: {case}/{scenario}")
            losses = [row["ncc_loss"] for row in candidates]
            edges = np.asarray(selection["proxy_edges"], float)
            if (selection["best"] != int(np.argmin(losses)) or not np.array_equal(selection["ncc_losses"], losses)
                    or edges.shape != (4,) or not np.isfinite(edges).all() or np.any(edges < 0)
                    or selection["proxy_case"] != float(edges.max())):
                raise ValueError(f"Invalid candidate selection/proxy: {case}/{scenario}")
            for weight in LAMBDA_GRID:
                row = records["B" if weight == 0 else f"lambda_{weight:g}"]
                if row.get("edge_weights") != [weight] * 4 or row.get("scales") != [3., 3.]:
                    raise ValueError(f"Prior weight/scale mismatch: {case}/{scenario}")
            risks.append([records["B" if weight == 0 else f"lambda_{weight:g}"]["metrics"]["surface_rms_tre_mm"] for weight in LAMBDA_GRID])
            scores.append(selection["proxy_case"])
            failures.append(int(records["B"]["metrics"]["case_failure_gt3mm"]))
            score_ids.append(case)
        included.append(case)
        clean.append(risks[0])
        combined.append(np.asarray(risks[1:]).T)
    if len(included) < 2:
        raise ValueError("Insufficient eligible independent development volumes")
    return dict(config=config, config_hash=config_hash, included=included, excluded=excluded,
                clean=np.asarray(clean), combined=np.asarray(combined), scores=scores,
                failures=failures, score_ids=score_ids, provenance=provenance)


def _write_locked(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Existing calibration differs; do not overwrite locked selection: {path}")
    atomic_json(path, value)


def calibrate(output_root=OUT, *, bootstrap_samples=10000):
    """Calibrate fixed C; finalize selection only when the adaptive gate fails."""
    root = Path(output_root)
    data = load_development(root)
    clean, combined, ids = data["clean"], data["combined"], data["included"]
    selection = select_lambda(clean, combined)
    risks = .5 * clean + .5 * combined.mean(axis=2)
    fold_estimates = []
    for index, validation in enumerate(data["config"]["split"]["development_folds"]):
        held = np.array([case in validation for case in ids])
        if not held.any() or np.count_nonzero(~held) < 2:
            raise ValueError("Every fold needs eligible validation and at least two training volumes")
        fitted = select_lambda(clean[~held], combined[~held])
        column = LAMBDA_GRID.index(fitted["selected_lambda"])
        fold_estimates.append(dict(fold=index, training_cases=np.asarray(ids)[~held].tolist(),
            validation_cases=np.asarray(ids)[held].tolist(), selected_lambda=fitted["selected_lambda"],
            validation_volume_risks=risks[held, column].tolist(),
            validation_mean_risk=float(risks[held, column].mean())))
    held_risks = [risk for fold in fold_estimates for risk in fold["validation_volume_risks"]]
    fixed = dict(status="FIXED_CALIBRATION_COMPLETE", config_hash=data["config_hash"], fixed_lambda=selection["selected_lambda"],
        selection=selection, n_assigned_development=25, included_cases=ids, excluded_cases=data["excluded"],
        folds=fold_estimates, train_only_selected_oof_mean_risk=float(np.mean(held_risks)),
        selection_basis="fixed-prior per-volume risks; no learned parameters on validation volumes",
        provenance=data["provenance"])
    gate = dict(config_hash=data["config_hash"], included_cases=ids, excluded_cases=data["excluded"],
                **proxy_gate(data["scores"], data["failures"], data["score_ids"], bootstrap_samples=bootstrap_samples))
    combined_rows = np.arange(len(data["scores"])) % len(SCENARIOS) != 0
    descriptive = proxy_gate(np.asarray(data["scores"])[combined_rows], np.asarray(data["failures"])[combined_rows],
                             np.asarray(data["score_ids"])[combined_rows], bootstrap_samples=bootstrap_samples)
    descriptive.pop("gate_passed")
    descriptive["role"] = "descriptive only; does not change pooled gate"
    gate["within_combined_descriptive"] = descriptive
    status = "ADAPTIVE_OOF_REQUIRED" if gate["gate_passed"] else "READY_FOR_FIXED_EVALUATION"
    final = dict(status=status, config_hash=data["config_hash"], fixed_lambda=selection["selected_lambda"],
                 adaptive=None, proxy_gate_passed=False, fixed_selection_hash=fingerprint(fixed), proxy_gate_hash=fingerprint(gate))
    out = root / "calibration"
    if gate["gate_passed"] and (out / "selection.json").exists():
        raise ValueError("Adaptive gate passed but a final selection already exists; do not reuse it")
    _write_locked(out / "fixed_selection.json", fixed)
    _write_locked(out / "proxy_gate.json", gate)
    with (out / "risk_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "fold", "lambda", "clean_rms", "combined_seed_mean_rms", "calibration_risk"])
        fold_map = {case: index for index, fold in enumerate(data["config"]["split"]["development_folds"]) for case in fold}
        for i, case in enumerate(ids):
            for j, weight in enumerate(LAMBDA_GRID):
                writer.writerow([case, fold_map[case], weight, clean[i, j], combined[i, j].mean(), risks[i, j]])
    with (out / "risk_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "mean_risk", "standard_error", "one_se_cutoff", "selected"])
        for weight, mean, se in zip(LAMBDA_GRID, selection["mean_risk"], selection["standard_error"]):
            writer.writerow([weight, mean, se, selection["one_se_cutoff"], weight == selection["selected_lambda"]])
    if not gate["gate_passed"]:
        _write_locked(out / "selection.json", final)
    return dict(status=status, fixed_lambda=selection["selected_lambda"], proxy_gate_passed=gate["gate_passed"],
                n_eligible=len(ids), n_excluded=len(data["excluded"]), config_hash=data["config_hash"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = calibrate(args.output)
    print(json.dumps(result, sort_keys=True))
    if result["proxy_gate_passed"]:
        print("Adaptive proxy gate passed: adaptive out-of-fold grid execution is required; final selection was NOT written.")


if __name__ == "__main__":
    main()
