"""Vertebra-wise motion recovery methods used by EXP-02/EXP-03.

The module keeps all three baselines in one physical-mm coordinate convention:

* ``independent_vertebrae``: Method B, one six-DoF motion per vertebra;
* ``rigid_global``: Method A, one six-DoF transform for the union mask;
* ``joint_soft``: Method C, all vertebra motions jointly optimized with a
  Lie-algebra spring penalty between adjacent motions.

The implementation is deliberately transparent and CPU friendly.  Images are
rendered with the calibrated cone-beam renderer and the same target is used by
all methods.  Lower pyramid levels use a detector-coordinate scaling of the
full-resolution 3x4 projection matrix, so downsampling does not silently
change the physical C-arm geometry.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .drr import CArmGeometry, Volume, ncc_loss, params_to_transform, render_cone_beam_drr


def translation_affine(shift: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(shift, dtype=np.float64).reshape(3)
    return out


def local_motion(params: Iterable[float], centre_mm: np.ndarray) -> np.ndarray:
    """Motion about a vertebral centre, with translations expressed in mm."""
    c = np.asarray(centre_mm, dtype=np.float64).reshape(3)
    return translation_affine(c) @ params_to_transform(params) @ translation_affine(-c)


def crop_volume(ct: np.ndarray, seg: np.ndarray, affine: np.ndarray, spacing: np.ndarray,
                labels: Iterable[int], margin: int, centre_shift: np.ndarray) -> Volume:
    mask = np.isin(seg, [int(v) for v in labels])
    coords = np.where(mask)
    if len(coords[0]) < 16:
        raise ValueError("requested labels have too few voxels")
    lo = np.maximum(0, np.array([c.min() for c in coords]) - int(margin))
    hi = np.minimum(seg.shape, np.array([c.max() + 1 for c in coords]) + int(margin))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    bone = np.clip(ct[slices] - 150.0, 0.0, 1800.0).astype(np.float32)
    data = bone * mask[slices].astype(np.float32)
    local_affine = np.asarray(affine, dtype=np.float64).copy()
    local_affine[:3, 3] = affine[:3, :3] @ lo.astype(np.float64) + affine[:3, 3]
    local_affine = translation_affine(-centre_shift) @ local_affine
    return Volume(data=data, spacing_mm=np.asarray(spacing, dtype=np.float32), affine_mm=local_affine)


def crop_component(ct: np.ndarray, seg: np.ndarray, affine: np.ndarray, spacing: np.ndarray,
                   label: int, margin: int, centre_shift: np.ndarray) -> tuple[Volume, np.ndarray]:
    mask = seg == int(label)
    coords = np.where(mask)
    if len(coords[0]) < 16:
        raise ValueError(f"label {label} has too few voxels")
    lo = np.maximum(0, np.array([c.min() for c in coords]) - int(margin))
    hi = np.minimum(seg.shape, np.array([c.max() + 1 for c in coords]) + int(margin))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local_mask = mask[slices]
    bone = np.clip(ct[slices] - 150.0, 0.0, 1800.0).astype(np.float32)
    data = bone * local_mask.astype(np.float32)
    local_affine = np.asarray(affine, dtype=np.float64).copy()
    local_affine[:3, 3] = affine[:3, :3] @ lo.astype(np.float64) + affine[:3, 3]
    local_affine = translation_affine(-centre_shift) @ local_affine
    points = np.c_[np.column_stack(coords).astype(np.float64), np.ones(len(coords[0]))]
    world = (affine @ points.T).T[:, :3]
    centre = world.mean(axis=0) - centre_shift
    return Volume(data=data, spacing_mm=np.asarray(spacing, dtype=np.float32), affine_mm=local_affine), centre


def load_motion_table(path: Path, labels: list[int]) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "motions" in payload:
        payload = payload["motions"]
    result: dict[int, np.ndarray] = {}
    for label in labels:
        item = payload.get(str(label), payload.get(label))
        if item is None:
            raise ValueError(f"motion JSON does not contain label {label}")
        if isinstance(item, dict) and "params" in item:
            item = item["params"]
        if isinstance(item, dict):
            values = [item.get(k, 0.0) for k in ("rx_deg", "ry_deg", "rz_deg", "tx_mm", "ty_mm", "tz_mm")]
        else:
            values = list(item)
        if len(values) != 6:
            raise ValueError(f"motion for label {label} must contain six values")
        result[int(label)] = np.r_[np.deg2rad(np.asarray(values[:3], dtype=np.float64)),
                                   np.asarray(values[3:], dtype=np.float64)]
    return result


def resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if tuple(image.shape) == tuple(shape):
        return image.copy()
    factors = (float(shape[0]) / image.shape[0], float(shape[1]) / image.shape[1])
    out = zoom(image, factors, order=1, prefilter=False)
    if out.shape != tuple(shape):
        fixed = np.zeros(shape, dtype=np.float32)
        h, w = min(shape[0], out.shape[0]), min(shape[1], out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return np.asarray(out, dtype=np.float32)


def scaled_projection(matrix: np.ndarray, full_shape: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    sy = float(shape[0]) / float(full_shape[0])
    sx = float(shape[1]) / float(full_shape[1])
    scale = np.diag([sx, sy, 1.0])
    return scale @ np.asarray(matrix, dtype=np.float64)


def roi(image: np.ndarray, pad: int = 10) -> tuple[slice, slice]:
    positive = image > (0.04 * float(image.max(initial=0.0)))
    ys, xs = np.where(positive)
    if len(xs) < 4:
        return slice(0, image.shape[0]), slice(0, image.shape[1])
    return (slice(max(0, int(ys.min()) - pad), min(image.shape[0], int(ys.max()) + pad + 1)),
            slice(max(0, int(xs.min()) - pad), min(image.shape[1], int(xs.max()) + pad + 1)))


def render(volume: Volume, params: np.ndarray, centre: np.ndarray, projection: np.ndarray,
           shape: tuple[int, int], sample_step_mm: float, max_samples: int) -> np.ndarray:
    return render_cone_beam_drr(
        volume, projection_matrix=projection, output_size=shape,
        transform_mm=local_motion(params, centre), sample_step_mm=sample_step_mm,
        max_samples=max_samples, chunk_rows=16, log_transform=False, normalize_output=False,
    )


def _bounds(max_rotation_deg: float, max_translation_mm: float) -> list[tuple[float, float]]:
    return [(-np.deg2rad(max_rotation_deg), np.deg2rad(max_rotation_deg))] * 3 + \
           [(-max_translation_mm, max_translation_mm)] * 3


def _pyramid_sizes(full_shape: tuple[int, int], requested: list[int] | None) -> list[tuple[int, int]]:
    max_side = min(full_shape)
    values = requested or [64, 128, 256]
    values = sorted({max(16, min(int(v), max_side)) for v in values})
    if max_side not in values:
        values.append(max_side)
    return [(int(v), int(round(v * full_shape[1] / full_shape[0]))) for v in values]


def _prepare_pyramids(target_raw: dict[str, np.ndarray], reference: dict[str, dict[int, np.ndarray]],
                      projections: dict[str, np.ndarray], full_shape: tuple[int, int],
                      levels: list[tuple[int, int]]) -> tuple[dict[str, dict[tuple[int, int], np.ndarray]],
                                                              dict[str, dict[int, dict[tuple[int, int], np.ndarray]]],
                                                              dict[str, dict[tuple[int, int], np.ndarray]]]:
    targets: dict[str, dict[tuple[int, int], np.ndarray]] = {v: {} for v in projections}
    refs: dict[str, dict[int, dict[tuple[int, int], np.ndarray]]] = {v: {} for v in projections}
    ps: dict[str, dict[tuple[int, int], np.ndarray]] = {v: {} for v in projections}
    for view, p in projections.items():
        for level in levels:
            targets[view][level] = resize_image(target_raw[view], level)
            ps[view][level] = scaled_projection(p, full_shape, level)
            for label, image in reference[view].items():
                refs[view].setdefault(label, {})[level] = resize_image(image, level)
    return targets, refs, ps


def motion_penalty(flat: np.ndarray, labels: list[int], sigma_rotation_deg: float,
                   sigma_translation_mm: float,
                   centres: list[np.ndarray] | None = None) -> float:
    params = np.asarray(flat, dtype=np.float64).reshape(len(labels), 6)
    wr = 1.0 / max(np.deg2rad(float(sigma_rotation_deg)), 1e-6)
    wt = 1.0 / max(float(sigma_translation_mm), 1e-6)
    values = []
    for i in range(len(labels) - 1):
        if centres is None:
            a = params_to_transform(params[i])
            b = params_to_transform(params[i + 1])
            delta = np.linalg.inv(a) @ b
        else:
            # M_i is the preoperative vertebral pose.  T_i is the global
            # transform induced by a local motion about that vertebral centre.
            # The penalty then measures Log(D_i^{-1} D'_i), where
            # D_i=M_i^{-1}M_{i+1} and D'_i=(T_iM_i)^{-1}(T_{i+1}M_{i+1}).
            mi = translation_affine(np.asarray(centres[i], dtype=np.float64))
            mj = translation_affine(np.asarray(centres[i + 1], dtype=np.float64))
            post_i = local_motion(params[i], np.asarray(centres[i], dtype=np.float64)) @ mi
            post_j = local_motion(params[i + 1], np.asarray(centres[i + 1], dtype=np.float64)) @ mj
            d_pre = np.linalg.inv(mi) @ mj
            d_post = np.linalg.inv(post_i) @ post_j
            delta = np.linalg.inv(d_pre) @ d_post
        omega = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
        theta = float(np.linalg.norm(omega))
        omega_hat = np.array([[0.0, -omega[2], omega[1]],
                              [omega[2], 0.0, -omega[0]],
                              [-omega[1], omega[0], 0.0]])
        if theta < 0.1:
            v_inv = np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0 + theta**2 / 720.0 + theta**4 / 30240.0) * (omega_hat @ omega_hat)
        else:
            coefficient = (1.0 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
            v_inv = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
        twist = np.r_[omega, v_inv @ delta[:3, 3]]
        rot_value = float(np.sum((twist[:3] * wr) ** 2))
        trans_value = float(np.sum((twist[3:] * wt) ** 2))
        values.append((rot_value, trans_value))
    if not values:
        return 0.0
    return float(np.asarray(values, dtype=np.float64).sum(axis=1).mean())


def _quantile_score(value: float, calibration: dict[str, object] | None,
                    key: str) -> float:
    """Map a raw uncertainty feature to [0, 1] using calibration Q10/Q90.

    Calibration is deliberately optional so old experiments retain their
    behaviour.  The accepted JSON shape is ``{"quantiles": {key: {"q10":
    ..., "q90": ...}}}``; a flat ``{key: {"q10": ..., "q90": ...}}`` is
    accepted too for small scripts.
    """
    raw = float(value)
    if not np.isfinite(raw):
        return 1.0
    if not calibration:
        return float(np.clip(raw / (1.0 + max(raw, 0.0)), 0.0, 1.0))
    quantiles = calibration.get("quantiles", calibration)
    item = quantiles.get(key, {}) if isinstance(quantiles, dict) else {}
    try:
        q10, q90 = float(item["q10"]), float(item["q90"])
    except (KeyError, TypeError, ValueError):
        return float(np.clip(raw / (1.0 + max(raw, 0.0)), 0.0, 1.0))
    if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10 + 1e-12:
        return float(np.clip(raw / (1.0 + max(raw, 0.0)), 0.0, 1.0))
    return float(np.clip((raw - q10) / (q90 - q10), 0.0, 1.0))


def uncertainty_calibration_from_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Estimate Q10/Q90 from independent-registration uncertainty rows.

    This helper is used by the cohort scripts and is intentionally based only
    on image-derived dispersion, never on ground-truth error.
    """
    values = {"translation_raw": [], "rotation_raw": []}
    for row in rows:
        for key, target in (("translation_uncertainty_raw", "translation_raw"),
                            ("rotation_uncertainty_raw", "rotation_raw")):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[target].append(value)
    quantiles = {}
    for key, seq in values.items():
        if seq:
            quantiles[key] = {"q10": float(np.quantile(seq, 0.10)),
                              "q90": float(np.quantile(seq, 0.90)),
                              "n": int(len(seq))}
        else:
            quantiles[key] = {"q10": 0.0, "q90": 1.0, "n": 0}
    return {"quantiles": quantiles, "n_rows": int(max(len(values["translation_raw"]), len(values["rotation_raw"]))),
            "source": "independent_image_only_dispersion", "quantile_definition": "empirical Q10/Q90"}


def _estimate_uncertainty(candidates: list[dict[str, object]], best_params: np.ndarray,
                          sigma_rotation_deg: float = 3.0,
                          sigma_translation_mm: float = 3.0,
                          calibration: dict[str, object] | None = None,
                          visibility_fraction: float = 1.0) -> dict[str, float]:
    """Estimate image-only uncertainty from the multi-start loss landscape.

    Candidate poses and their final AP/LAT losses are produced by the
    independent registration stage.  No ground-truth pose is used here.
    The bounded score is a mixture of normalized pose spread, loss entropy,
    and the best image residual; larger values mean lower confidence.
    """
    visibility_fraction = float(np.clip(visibility_fraction, 0.0, 1.0))
    if not candidates:
        return {"uncertainty_score": 1.0, "translation_uncertainty_score": 1.0,
                "rotation_uncertainty_score": 1.0, "translation_uncertainty_raw": 1.0,
                "rotation_uncertainty_raw": 1.0, "image_confidence": 0.0, "translation_dispersion_mm": 0.0,
                "rotation_dispersion_deg": 0.0, "candidate_entropy": 1.0,
                "candidate_loss_gap": 0.0, "best_loss": 1.0, "uncertainty_source_loss": 1.0,
                "n_candidates": 0, "normalized_spread": 1.0,
                "visibility_fraction": visibility_fraction}
    poses = np.asarray([np.asarray(c["params"], dtype=np.float64) for c in candidates])
    losses = np.asarray([float(c["loss"]) for c in candidates], dtype=np.float64)
    finite = np.isfinite(losses)
    if not np.all(finite):
        losses = np.where(finite, losses, np.nanmax(losses[finite]) + 1.0 if np.any(finite) else 1.0)
    temperature = max(float(np.std(losses)), 1e-3)
    logits = -(losses - float(np.min(losses))) / temperature
    logits -= float(np.max(logits))
    probs = np.exp(logits)
    probs /= max(float(np.sum(probs)), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))) /
                    max(np.log(max(len(probs), 2)), 1e-12))
    scale = np.r_[np.full(3, max(np.deg2rad(float(sigma_rotation_deg)), 1e-6)),
                  np.full(3, max(float(sigma_translation_mm), 1e-6))]
    normalized = poses / scale
    centre = np.sum(normalized * probs[:, None], axis=0)
    covariance = (normalized - centre).T @ ((normalized - centre) * probs[:, None])
    spread = float(np.sqrt(max(np.trace(covariance) / 6.0, 0.0)))
    spread_score = float(np.clip(spread / (1.0 + spread), 0.0, 1.0))
    best_index = int(np.argmin(losses))
    best = poses[best_index]
    trans_disp = float(np.sqrt(np.sum(probs * np.sum((poses[:, 3:] - best[3:]) ** 2, axis=1))))
    rot_disp = float(np.sqrt(np.sum(probs * np.sum((poses[:, :3] - best[:3]) ** 2, axis=1))))
    ordered = np.sort(losses)
    gap = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
    # NCC loss is 1-NCC and can reach 2.  Use a calibrated sigmoid around a
    # pilot failure threshold (0.6) so a poor AP/LAT match dominates the
    # uncertainty score instead of being diluted by averaging.
    residual_midpoint, residual_temperature = 0.6, 0.1
    residual_score = float(1.0 / (1.0 + np.exp(-(float(losses[best_index]) - residual_midpoint) / residual_temperature)))
    # Entropy alone is high even when all starts converge to the same pose.
    # Gate it by pose/loss dispersion so stable convergence is low uncertainty.
    loss_gate = float(np.clip(np.std(losses) / 0.05, 0.0, 1.0))
    effective_entropy = float(entropy * max(spread_score, loss_gate))
    # Separate uncertainty channels are useful because translations and
    # rotations have different observability in AP/LAT projections.  A poor
    # residual raises both channels, while the ensemble spread remains
    # channel-specific.
    translation_raw = float(0.5 * (trans_disp / max(float(sigma_translation_mm), 1e-6)) + 0.5 * residual_score)
    rotation_raw = float(0.5 * (rot_disp / max(np.deg2rad(float(sigma_rotation_deg)), 1e-6)) + 0.5 * residual_score)
    translation_score = _quantile_score(translation_raw, calibration, "translation_raw")
    rotation_score = _quantile_score(rotation_raw, calibration, "rotation_raw")
    # Retain the historical scalar score for reports and isotropic callers.
    score = float(np.clip(0.5 * (translation_score + rotation_score), 0.0, 1.0))
    if spread < 1e-3 and losses[best_index] >= 0.8:
        score = 1.0
        translation_score = rotation_score = 1.0
    return {"uncertainty_score": score,
            "translation_uncertainty_score": float(translation_score),
            "rotation_uncertainty_score": float(rotation_score),
            "translation_uncertainty_raw": translation_raw,
            "rotation_uncertainty_raw": rotation_raw,
            "image_confidence": float(1.0 - score), "uncertainty_source_loss": float(losses[best_index]), "translation_dispersion_mm": trans_disp,
            "rotation_dispersion_deg": float(np.degrees(rot_disp)),
            "candidate_entropy": entropy, "candidate_loss_gap": gap,
            "best_loss": float(losses[best_index]), "n_candidates": int(len(candidates)),
            "normalized_spread": spread, "visibility_fraction": visibility_fraction}


def _independent_estimate(labels, components, gt, targets, refs, ps, levels, full_shape,
                          maxiter, max_translation_mm, max_rotation_deg, sample_step_mm,
                          max_samples, seed, multi_start, init_translation_mm, init_rotation_deg,
                          uncertainty_calibration: dict[str, object] | None = None):
    estimates: dict[int, np.ndarray] = {}
    all_predictions: dict[str, dict[tuple[int, int], dict[int, np.ndarray]]] = {
        v: {level: {} for level in levels} for v in ps
    }
    rows: list[dict[str, object]] = []
    bounds = _bounds(max_rotation_deg, max_translation_mm)
    rng = np.random.default_rng(seed)
    for label_index, label in enumerate(labels):
        volume, centre = components[label]
        residuals = {}
        for view in ps:
            residuals[view] = {}
            for level in levels:
                residuals[view][level] = targets[view][level].copy()
                for previous in estimates:
                    residuals[view][level] -= all_predictions[view][level][previous]
        rois = {view: {level: roi(refs[view][label][level]) for level in levels} for view in ps}

        starts = [np.zeros(6, dtype=np.float64)]
        for _ in range(max(0, int(multi_start) - 1)):
            starts.append(np.r_[np.deg2rad(rng.uniform(-init_rotation_deg, init_rotation_deg, 3)),
                                rng.uniform(-init_translation_mm, init_translation_mm, 3)])
        best_params, best_loss, best_ncc, best_evals = None, float("inf"), -float("inf"), 0
        candidates: list[dict[str, object]] = []
        label_start = time.perf_counter()
        for start in starts:
            params = np.asarray(start, dtype=np.float64).copy()
            total_evals = 0
            for level in levels:
                def objective(candidate: np.ndarray) -> float:
                    losses = []
                    for view, p in ps.items():
                        pred = render(volume, candidate, centre, p[level], level, sample_step_mm, max_samples)
                        ys, xs = rois[view][level]
                        losses.append(0.85 * ncc_loss(residuals[view][level][ys, xs], pred[ys, xs]) +
                                      0.15 * ncc_loss(residuals[view][level], pred))
                    return float(np.mean(losses))
                result = minimize(objective, params, method="Powell", bounds=bounds,
                                  options={"maxiter": int(maxiter), "xtol": 2e-3, "ftol": 2e-3})
                params = np.asarray(result.x, dtype=np.float64)
                total_evals += int(getattr(result, "nfev", 0))
            full_losses = []
            for view, p in ps.items():
                pred = render(volume, params, centre, p[levels[-1]], levels[-1], sample_step_mm, max_samples)
                full_losses.append(ncc_loss(residuals[view][levels[-1]], pred))
            loss = float(np.mean(full_losses))
            candidates.append({"params": params.copy(), "loss": loss, "n_evaluations": total_evals})
            if loss < best_loss:
                best_params, best_loss, best_ncc, best_evals = params.copy(), loss, 1.0 - loss, total_evals
        assert best_params is not None
        visibility_values = []
        for view in ps:
            image = np.asarray(refs[view][label][levels[-1]], dtype=np.float32)
            threshold = 0.04 * float(image.max(initial=0.0))
            visibility_values.append(float(np.count_nonzero(image > threshold)) / max(image.size, 1))
        # A vertebra is observable if it occupies pixels in at least one view;
        # the maximum is used because AP and LAT can have very different FOVs.
        visibility_fraction = max(visibility_values, default=0.0)
        uncertainty = _estimate_uncertainty(candidates, best_params,
                                            calibration=uncertainty_calibration,
                                            visibility_fraction=visibility_fraction)
        estimates[label] = best_params
        for view, p in ps.items():
            for level in levels:
                all_predictions[view][level][label] = render(volume, best_params, centre, p[level], level, sample_step_mm, max_samples)
        diff = best_params - gt[label]
        rows.append({
            "label": int(label), "ground_truth_rx_ry_rz_tx_ty_tz_mm": gt[label].tolist(),
            "estimated_rx_ry_rz_tx_ty_tz_mm": best_params.tolist(),
            "estimated_params_rx_ry_rz_tx_ty_tz_mm": best_params.tolist(),
            "translation_error_mm": float(np.linalg.norm(diff[3:])),
            "rotation_vector_error_deg": float(np.degrees(Rotation.from_matrix(params_to_transform(estimates[label])[:3, :3].T @ params_to_transform(gt[label])[:3, :3]).magnitude())),
            "final_loss": float(best_loss), "best_ncc": float(best_ncc),
            "n_starts": int(len(starts)), "n_evaluations": int(best_evals),
            "runtime": float(time.perf_counter() - label_start), "pose_mode": "local",
            **uncertainty,
        })
    uncertainty_by_label = {
        int(row["label"]): {key: row[key] for key in (
            "uncertainty_score", "image_confidence", "translation_dispersion_mm", "rotation_dispersion_deg",
            "candidate_entropy", "candidate_loss_gap", "best_loss", "uncertainty_source_loss", "n_candidates",
            "normalized_spread", "visibility_fraction", "translation_uncertainty_score", "rotation_uncertainty_score",
            "translation_uncertainty_raw", "rotation_uncertainty_raw")}
        for row in rows
    }
    return estimates, all_predictions, rows, uncertainty_by_label


def _joint_soft_estimate(labels, components, gt, targets, ps, levels, initial, maxiter,
                         max_translation_mm, max_rotation_deg, sample_step_mm, max_samples,
                         lambda_motion, sigma_rotation_deg, sigma_translation_mm,
                         centres: list[np.ndarray] | None = None):
    params = np.concatenate([initial[label] for label in labels]).astype(np.float64)
    bounds = _bounds(max_rotation_deg, max_translation_mm) * len(labels)
    history: list[float] = []
    level_results = []
    for level in levels:
        def objective(flat: np.ndarray) -> float:
            flat = np.asarray(flat, dtype=np.float64)
            pose = flat.reshape(len(labels), 6)
            image_losses = []
            for view, p in ps.items():
                prediction = np.zeros(level, dtype=np.float32)
                for j, label in enumerate(labels):
                    volume, centre = components[label]
                    prediction += render(volume, pose[j], centre, p[level], level, sample_step_mm, max_samples)
                image_losses.append(ncc_loss(targets[view][level], prediction))
            penalty = motion_penalty(flat, labels, sigma_rotation_deg, sigma_translation_mm,
                                     centres=centres)
            value = float(np.mean(image_losses) + float(lambda_motion) * penalty)
            history.append(value)
            return value
        # Coordinate descent over six-parameter vertebral blocks.  The
        # optimized objective is still the full 30-D L_total; updating one
        # block at a time avoids the very large finite-search overhead of a
        # black-box Powell sweep over all 30 variables on a CPU renderer.
        total_evals = 0
        for _sweep in range(max(1, int(maxiter))):
            for j, label in enumerate(labels):
                pose = params.reshape(len(labels), 6)
                current_predictions: dict[str, np.ndarray] = {}
                current_component: dict[str, np.ndarray] = {}
                for view, p in ps.items():
                    current_component[view] = render(components[label][0], pose[j], components[label][1], p[level], level, sample_step_mm, max_samples)
                    total_prediction = np.zeros(level, dtype=np.float32)
                    for k, other_label in enumerate(labels):
                        if other_label == label:
                            total_prediction += current_component[view]
                        else:
                            total_prediction += render(components[other_label][0], pose[k], components[other_label][1], p[level], level, sample_step_mm, max_samples)
                    current_predictions[view] = total_prediction

                def block_objective(block: np.ndarray) -> float:
                    candidate_losses = []
                    for view, p in ps.items():
                        candidate_component = render(components[label][0], block, components[label][1], p[level], level, sample_step_mm, max_samples)
                        prediction = current_predictions[view] - current_component[view] + candidate_component
                        candidate_losses.append(ncc_loss(targets[view][level], prediction))
                    candidate_flat = params.copy()
                    candidate_flat[j * 6:(j + 1) * 6] = block
                    value = float(np.mean(candidate_losses) + float(lambda_motion) * motion_penalty(candidate_flat, labels, sigma_rotation_deg, sigma_translation_mm, centres=centres))
                    history.append(value)
                    return value

                block_result = minimize(block_objective, pose[j].copy(), method="Powell", bounds=bounds[j * 6:(j + 1) * 6], options={"maxiter": 1, "xtol": 2e-3, "ftol": 2e-3})
                params[j * 6:(j + 1) * 6] = np.asarray(block_result.x, dtype=np.float64)
                total_evals += int(getattr(block_result, "nfev", 0))
        final_loss = float(objective(params))
        level_results.append({"level": list(level), "loss": final_loss, "nfev": int(total_evals)})
    return {label: params[i * 6:(i + 1) * 6].copy() for i, label in enumerate(labels)}, level_results, history


def _global_estimate(labels, components, gt, global_volume, centres, targets, ps, levels,
                     maxiter, max_translation_mm, max_rotation_deg, sample_step_mm, max_samples):
    params = np.zeros(6, dtype=np.float64)
    bounds = _bounds(max_rotation_deg, max_translation_mm)
    level_results = []
    for level in levels:
        def objective(candidate: np.ndarray) -> float:
            losses = []
            for view, p in ps.items():
                pred = render(global_volume, candidate, np.zeros(3), p[level], level, sample_step_mm, max_samples)
                losses.append(ncc_loss(targets[view][level], pred))
            return float(np.mean(losses))
        result = minimize(objective, params, method="Powell", bounds=bounds,
                          options={"maxiter": int(maxiter), "xtol": 2e-3, "ftol": 2e-3})
        params = np.asarray(result.x, dtype=np.float64)
        level_results.append({"level": list(level), "loss": float(result.fun), "nfev": int(getattr(result, "nfev", 0))})
    return params, level_results


def run_method(case_dir: Path, output: Path, labels: list[int], size: tuple[int, int], seed: int,
               maxiter: int, max_translation_mm: float, max_rotation_deg: float,
               sample_step_mm: float, max_samples: int, motion_json: Path,
               method: str = "independent_vertebrae", lambda_motion: float = 0.0,
               pyramid_sizes: list[int] | None = None, multi_start: int = 3,
               init_translation_mm: float = 3.0, init_rotation_deg: float = 3.0,
               sigma_rotation_deg: float = 3.0, sigma_translation_mm: float = 3.0,
               uncertainty_calibration: dict[str, object] | None = None,
               target_override_dir: Path | None = None,
               views: Iterable[str] | None = None) -> dict:
    case_dir = Path(case_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ct_img = nib.load(str(case_dir / "ct.nii.gz"))
    seg_img = nib.load(str(case_dir / "seg.nii.gz"))
    ct, seg = np.asarray(ct_img.dataobj, dtype=np.float32), np.asarray(seg_img.dataobj)
    if ct.ndim != 3 or seg.ndim != 3:
        raise ValueError(f"CT and segmentation must be 3-D; got {ct.shape} and {seg.shape}")
    if ct.shape != seg.shape:
        # PPCNet contains occasional all-zero segmentation padding planes.
        # Align those grids in voxel coordinates when their physical affines
        # agree, preserving the CT grid used by the renderer.
        if not np.allclose(np.asarray(ct_img.affine), np.asarray(seg_img.affine), atol=1e-3):
            raise ValueError(f"CT and segmentation shapes differ and affines do not align; got {ct.shape} and {seg.shape}")
        common = tuple(min(int(a), int(b)) for a, b in zip(ct.shape, seg.shape))
        extra_nonzero = False
        for axis in range(3):
            if seg.shape[axis] > common[axis]:
                sl = [slice(None)] * 3
                sl[axis] = slice(common[axis], int(seg.shape[axis]))
                extra_nonzero = extra_nonzero or bool(np.any(seg[tuple(sl)] != 0))
        if extra_nonzero:
            raise ValueError(f"CT and segmentation shapes differ with nonzero segmentation outside CT grid; got {ct.shape} and {seg.shape}")
        aligned = np.zeros(ct.shape, dtype=seg.dtype)
        aligned[tuple(slice(0, n) for n in common)] = seg[tuple(slice(0, n) for n in common)]
        seg = aligned
    spacing, affine = np.asarray(ct_img.header.get_zooms()[:3], dtype=np.float32), np.asarray(ct_img.affine, dtype=np.float64)
    all_idx = np.argwhere(np.isin(seg, labels))
    all_world = (affine @ np.c_[all_idx, np.ones(len(all_idx))].T).T[:, :3]
    centre_shift = all_world.mean(axis=0)
    shape = tuple(map(int, size))
    ap_geometry = CArmGeometry(detector_size_px=shape, pixel_spacing_mm=(1.0, 1.0), source_to_isocenter_mm=750.0, source_to_detector_mm=1200.0, view_angle_deg=90.0, rotation_axis="x")
    lat_geometry = CArmGeometry(detector_size_px=shape, pixel_spacing_mm=(1.0, 1.0), source_to_isocenter_mm=750.0, source_to_detector_mm=1200.0, view_angle_deg=90.0, rotation_axis="y")
    # Select the observations used by the image loss.  The default remains
    # the original AP+LAT acquisition; EXP-02 passes AP or LAT to perform a
    # genuine single-view ablation rather than masking one view afterward.
    requested_views = [str(v).upper() for v in views] if views is not None else ["AP", "LAT"]
    unknown_views = sorted(set(requested_views) - {"AP", "LAT"})
    if unknown_views or not requested_views:
        raise ValueError(f"views must contain AP and/or LAT; got {requested_views}")
    projections_all = {"AP": ap_geometry.projection_matrix(), "LAT": lat_geometry.projection_matrix()}
    projections = {name: projections_all[name] for name in ("AP", "LAT") if name in requested_views}
    components = {int(label): crop_component(ct, seg, affine, spacing, int(label), 4, centre_shift) for label in labels}
    gt = load_motion_table(Path(motion_json), labels)
    reference = {view: {} for view in projections}
    target_raw = {view: np.zeros(shape, dtype=np.float32) for view in projections}
    for label in labels:
        volume, centre = components[int(label)]
        for view, p in projections.items():
            reference[view][int(label)] = render(volume, np.zeros(6), centre, p, shape, sample_step_mm, max_samples)
            target_raw[view] += render(volume, gt[int(label)], centre, p, shape, sample_step_mm, max_samples)
    # Robustness experiments may replace the synthetic target with a
    # deterministically perturbed AP/LAT image (noise, occlusion, or a view
    # perturbation).  Keep target generation as the default so existing
    # experiments remain unchanged.  The override is intentionally applied
    # after rendering and before pyramid construction, which means both
    # All methods see exactly the same perturbed observation.
    if target_override_dir is not None:
        override_root = Path(target_override_dir)
        for view in projections:
            candidates = (
                override_root / view / "target.npy",
                override_root / f"{view}.npy",
                override_root / f"target_{view}.npy",
            )
            source = next((path for path in candidates if path.exists()), None)
            if source is None:
                raise FileNotFoundError(
                    f"target override for {view} not found under {override_root}; "
                    "expected AP/target.npy, LAT/target.npy, or AP.npy/LAT.npy"
                )
            override = np.asarray(np.load(source), dtype=np.float32)
            if override.shape != shape:
                raise ValueError(
                    f"target override {source} has shape {override.shape}, expected {shape}"
                )
            target_raw[view] = override
    levels = _pyramid_sizes(shape, pyramid_sizes)
    targets, refs, ps = _prepare_pyramids(target_raw, reference, projections, shape, levels)
    start_time = time.perf_counter()
    if method == "independent_vertebrae":
        estimates, predictions, rows, uncertainty_by_label = _independent_estimate(labels, components, gt, targets, refs, ps, levels, shape, maxiter, max_translation_mm, max_rotation_deg, sample_step_mm, max_samples, seed, multi_start, init_translation_mm, init_rotation_deg, uncertainty_calibration)
        joint_meta = {"uncertainty_by_label": uncertainty_by_label}
    elif method == "rigid_global":
        global_volume = crop_volume(ct, seg, affine, spacing, labels, 4, centre_shift)
        centres = {label: components[label][1] for label in labels}
        global_params, level_results = _global_estimate(labels, components, gt, global_volume, centres, targets, ps, levels, maxiter, max_translation_mm, max_rotation_deg, sample_step_mm, max_samples)
        estimates = {label: global_params.copy() for label in labels}
        predictions = {view: {level: {} for level in levels} for view in projections}
        rows = []
        for label in labels:
            centre = centres[label]
            true_center = (local_motion(gt[label], centre) @ np.r_[centre, 1.0])[:3]
            pred_center = (params_to_transform(global_params) @ np.r_[centre, 1.0])[:3]
            delta_rot = Rotation.from_matrix(params_to_transform(global_params)[:3, :3].T @ params_to_transform(gt[label])[:3, :3]).as_rotvec()
            diff = global_params - gt[label]
            rows.append({"label": int(label), "ground_truth_rx_ry_rz_tx_ty_tz_mm": gt[label].tolist(), "estimated_rx_ry_rz_tx_ty_tz_mm": global_params.tolist(), "estimated_params_rx_ry_rz_tx_ty_tz_mm": global_params.tolist(), "translation_error_mm": float(np.linalg.norm(pred_center - true_center)), "rotation_vector_error_deg": float(np.degrees(np.linalg.norm(delta_rot))), "parameter_translation_error_mm": float(np.linalg.norm(diff[3:])), "parameter_rotation_error_deg": float(np.degrees(Rotation.from_matrix(params_to_transform(estimates[label])[:3, :3].T @ params_to_transform(gt[label])[:3, :3]).magnitude())), "final_loss": float(level_results[-1]["loss"]), "best_ncc": float(1.0 - level_results[-1]["loss"]), "n_starts": 1, "n_evaluations": int(sum(v["nfev"] for v in level_results)), "runtime": float(time.perf_counter() - start_time) / max(len(labels), 1), "pose_mode": "global"})
        joint_meta = {"global_params": global_params.tolist(), "levels": level_results}
    elif method == "joint_soft":
        initial, _, initial_rows, uncertainty_by_label = _independent_estimate(labels, components, gt, targets, refs, ps, levels, shape, maxiter, max_translation_mm, max_rotation_deg, sample_step_mm, max_samples, seed, multi_start, init_translation_mm, init_rotation_deg, uncertainty_calibration)
        if float(lambda_motion) == 0.0:
            estimates = initial
            rows = initial_rows
            joint_meta = {"lambda": 0.0, "initialized_from": "independent_vertebrae", "skipped_joint_refinement": True, "uncertainty_by_label": uncertainty_by_label}
            predictions = {view: {level: {} for level in levels} for view in projections}
        else:
            motion_centres = [components[int(label)][1] for label in labels]
            estimates, level_results, history = _joint_soft_estimate(labels, components, gt, targets, ps, levels, initial, maxiter, max_translation_mm, max_rotation_deg, sample_step_mm, max_samples, lambda_motion, sigma_rotation_deg, sigma_translation_mm, centres=motion_centres)
            rows = []
            for index, label in enumerate(labels):
                diff = estimates[label] - gt[label]
                row = {"label": int(label), "ground_truth_rx_ry_rz_tx_ty_tz_mm": gt[label].tolist(), "estimated_rx_ry_rz_tx_ty_tz_mm": estimates[label].tolist(), "estimated_params_rx_ry_rz_tx_ty_tz_mm": estimates[label].tolist(), "translation_error_mm": float(np.linalg.norm(diff[3:])), "rotation_vector_error_deg": float(np.degrees(Rotation.from_matrix(params_to_transform(estimates[label])[:3, :3].T @ params_to_transform(gt[label])[:3, :3]).magnitude())), "final_loss": float(level_results[-1]["loss"]), "best_ncc": float(1.0 - level_results[-1]["loss"]), "n_starts": int(multi_start), "n_evaluations": int(sum(v["nfev"] for v in level_results)), "runtime": float(time.perf_counter() - start_time) / max(len(labels), 1), "pose_mode": "local", **uncertainty_by_label[int(label)]}
                rows.append(row)
            joint_meta = {"lambda": float(lambda_motion), "sigma_rotation_deg": float(sigma_rotation_deg), "sigma_translation_mm": float(sigma_translation_mm), "initialized_from": "independent_vertebrae", "motion_penalty_definition": "preoperative_relative_pose_corrected_se3_log", "levels": level_results, "motion_penalty_final": motion_penalty(np.concatenate([estimates[l] for l in labels]), labels, sigma_rotation_deg, sigma_translation_mm, centres=motion_centres), "history_length": len(history), "uncertainty_by_label": uncertainty_by_label}
            predictions = {view: {level: {} for level in levels} for view in projections}
    else:
        raise ValueError(f"unknown method: {method}")
    # Save final prediction images for visual checking.
    recovered = {view: np.zeros(shape, dtype=np.float32) for view in projections}
    for label in labels:
        volume, centre = components[label]
        for view, p in projections.items():
            if method == "rigid_global":
                recovered[view] += render_cone_beam_drr(volume, projection_matrix=p, output_size=shape,
                    transform_mm=params_to_transform(estimates[label]), sample_step_mm=sample_step_mm,
                    max_samples=max_samples, log_transform=False, normalize_output=False)
            else:
                recovered[view] += render(volume, estimates[label], centre, p, shape, sample_step_mm, max_samples)
    for view in projections:
        np.save(output / f"target_{view}.npy", target_raw[view])
        np.save(output / f"recovered_{view}.npy", recovered[view])
        np.save(output / f"difference_{view}.npy", np.abs(target_raw[view] - recovered[view]))
    summary = {"case_dir": str(case_dir.resolve()), "output": str(output.resolve()), "method": method, "lambda_motion": float(lambda_motion), "labels": labels, "views": list(projections), "output_shape": list(shape), "pyramid_sizes": [list(v) for v in levels], "coordinate_units": "physical_mm", "renderer": "cone_beam_physical_mm", "projection_matrices": {v: p.tolist() for v, p in projections.items()}, "virtual_carm": {"AP": ap_geometry.to_dict(), "LAT": lat_geometry.to_dict()}, "ground_truth_poses": {str(k): v.tolist() for k, v in gt.items()}, "per_vertebra": rows, "mean_translation_error_mm": float(np.mean([r["translation_error_mm"] for r in rows])), "median_translation_error_mm": float(np.median([r["translation_error_mm"] for r in rows])), "mean_rotation_vector_error_deg": float(np.mean([r["rotation_vector_error_deg"] for r in rows])), "median_rotation_vector_error_deg": float(np.median([r["rotation_vector_error_deg"] for r in rows])), "runtime": float(time.perf_counter() - start_time), "motion_json": str(Path(motion_json).resolve()), "target_override_dir": str(Path(target_override_dir).resolve()) if target_override_dir is not None else None, "optimizer": {"multi_start": int(multi_start), "maxiter_per_level": int(maxiter), "max_translation_mm": float(max_translation_mm), "max_rotation_deg": float(max_rotation_deg), "init_translation_mm": float(init_translation_mm), "init_rotation_deg": float(init_rotation_deg), "pyramid_sizes": [list(v) for v in levels]}, "joint_meta": joint_meta}
    (output / "motion_demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
