"""Anatomy-aware metrics for vertebra-wise motion recovery.

The usual per-vertebra translation/rotation error can make a global rigid
registration look deceptively good when the spine undergoes intervertebral
motion.  This module evaluates the quantities that a global transform cannot
represent: adjacent relative-pose error and change in the distance between
adjacent vertebral centres.

All transforms use the project convention (millimetres and an ``[rx, ry, rz,
tx, ty, tz]`` parameter vector, with the first three entries in radians for
the internal representation).  A pre-operative vertebral frame is represented
by a translation to its CT centroid.  For Methods B/C the estimated motion is
applied locally about that centroid; for Method A the estimated transform is
applied globally, exactly as in ``motion_methods._global_estimate``.
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .drr import params_to_transform
from .motion_methods import local_motion, translation_affine


def params_from_pose_record(value: object) -> np.ndarray:
    """Parse a CSV JSON pose record into ``[rx, ry, rz, tx, ty, tz]``.

    ``ground_truth_pose`` and ``estimated_pose`` are JSON objects with degree
    angles.  For convenience, six-element sequences in the internal radians
    convention are accepted as well.
    """
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        vals = [value.get(k, 0.0) for k in
                ("rx_deg", "ry_deg", "rz_deg", "tx_mm", "ty_mm", "tz_mm")]
        return np.r_[np.deg2rad(np.asarray(vals[:3], dtype=np.float64)),
                     np.asarray(vals[3:], dtype=np.float64)]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 6:
        raise ValueError("a pose must contain six values")
    return arr.copy()


def preoperative_frame(centre_mm: Iterable[float]) -> np.ndarray:
    """Return the pre-operative frame ``M_i`` for a CT centroid."""
    return translation_affine(np.asarray(centre_mm, dtype=np.float64).reshape(3))


def _postoperative_frame(params: np.ndarray, centre_mm: np.ndarray,
                         method: str, *, ground_truth: bool = False) -> np.ndarray:
    """Apply one motion to a vertebral frame using the pipeline convention."""
    m = preoperative_frame(centre_mm)
    # Ground truth is generated as a local motion for every vertebra.  A
    # rigid_global estimate, however, is one transform applied to the whole
    # union mask; treating it as a local motion would manufacture relative
    # motion and invalidate the Method-A comparison.
    is_global = (not ground_truth) and (method == "rigid_global" or
                                        method.startswith("rigid_global"))
    motion = params_to_transform(params) if is_global else local_motion(params, centre_mm)
    return motion @ m


def _se3_log_translation(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation vector and the SE(3) logarithm translation component."""
    delta = np.asarray(transform, dtype=np.float64)
    omega = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(omega))
    omega_hat = np.array([[0.0, -omega[2], omega[1]],
                          [omega[2], 0.0, -omega[0]],
                          [-omega[1], omega[0], 0.0]])
    if theta < 1e-8:
        v_inv = np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0) * (omega_hat @ omega_hat)
    else:
        # Stable away from pi for the small surgical motions used here.
        sin_theta = float(np.sin(theta))
        if abs(sin_theta) < 1e-8:
            coefficient = 1.0 / (theta * theta)
        else:
            coefficient = (1.0 / (theta ** 2)) - ((1.0 + np.cos(theta)) /
                                                  (2.0 * theta * sin_theta))
        v_inv = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
    return omega, v_inv @ delta[:3, 3]


def relative_pose_error(gt_i: Iterable[float], gt_j: Iterable[float],
                        est_i: Iterable[float], est_j: Iterable[float],
                        centre_i_mm: Iterable[float], centre_j_mm: Iterable[float],
                        method: str = "independent_vertebrae") -> dict[str, float]:
    """Compute adjacent relative-pose error for one vertebra pair.

    Let ``D_pre=M_i^-1 M_j`` and ``D_post=(T_i M_i)^-1(T_j M_j)``.  The
    returned error is the SE(3) logarithm of ``D_gt_post^-1 D_est_post``;
    rotation is reported in degrees and translation in millimetres.  The
    pre-operative relative pose is therefore removed before comparison.
    """
    ci = np.asarray(centre_i_mm, dtype=np.float64).reshape(3)
    cj = np.asarray(centre_j_mm, dtype=np.float64).reshape(3)
    pgi, pgj = params_from_pose_record(gt_i), params_from_pose_record(gt_j)
    pei, pej = params_from_pose_record(est_i), params_from_pose_record(est_j)
    mi, mj = preoperative_frame(ci), preoperative_frame(cj)
    d_pre = np.linalg.inv(mi) @ mj
    gt_i_post = _postoperative_frame(pgi, ci, method, ground_truth=True)
    gt_j_post = _postoperative_frame(pgj, cj, method, ground_truth=True)
    est_i_post = _postoperative_frame(pei, ci, method, ground_truth=False)
    est_j_post = _postoperative_frame(pej, cj, method, ground_truth=False)
    d_gt = np.linalg.inv(gt_i_post) @ gt_j_post
    d_est = np.linalg.inv(est_i_post) @ est_j_post
    err = np.linalg.inv(d_gt) @ d_est
    omega, v = _se3_log_translation(err)
    gt_omega, gt_v = _se3_log_translation(np.linalg.inv(d_pre) @ d_gt)
    est_omega, est_v = _se3_log_translation(np.linalg.inv(d_pre) @ d_est)
    return {
        "relative_translation_error_mm": float(np.linalg.norm(v)),
        "relative_rotation_error_deg": float(np.rad2deg(np.linalg.norm(omega))),
        "gt_relative_translation_mm": float(np.linalg.norm(gt_v)),
        "gt_relative_rotation_deg": float(np.rad2deg(np.linalg.norm(gt_omega))),
        "estimated_relative_translation_mm": float(np.linalg.norm(est_v)),
        "estimated_relative_rotation_deg": float(np.rad2deg(np.linalg.norm(est_omega))),
        "preoperative_distance_mm": float(np.linalg.norm(cj - ci)),
    }


def intervertebral_distance_error(gt_i: Iterable[float], gt_j: Iterable[float],
                                  est_i: Iterable[float], est_j: Iterable[float],
                                  centre_i_mm: Iterable[float], centre_j_mm: Iterable[float],
                                  method: str = "independent_vertebrae") -> dict[str, float]:
    """Return error in the *change* of adjacent-centre distance.

    ``distance_change_error = |(d_est_post-d_pre) - (d_gt_post-d_pre)|``.  The
    signed changes are retained so a reviewer can distinguish compression
    from separation.  Method A uses its global transform, while B/C use
    vertebra-local transforms.
    """
    ci = np.asarray(centre_i_mm, dtype=np.float64).reshape(3)
    cj = np.asarray(centre_j_mm, dtype=np.float64).reshape(3)
    pgi, pgj = params_from_pose_record(gt_i), params_from_pose_record(gt_j)
    pei, pej = params_from_pose_record(est_i), params_from_pose_record(est_j)
    mi, mj = preoperative_frame(ci), preoperative_frame(cj)
    pre_i = (mi @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    pre_j = (mj @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    gt_i_post = (_postoperative_frame(pgi, ci, method, ground_truth=True) @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    gt_j_post = (_postoperative_frame(pgj, cj, method, ground_truth=True) @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    est_i_post = (_postoperative_frame(pei, ci, method, ground_truth=False) @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    est_j_post = (_postoperative_frame(pej, cj, method, ground_truth=False) @ np.r_[0.0, 0.0, 0.0, 1.0])[:3]
    d_pre = float(np.linalg.norm(pre_j - pre_i))
    d_gt = float(np.linalg.norm(gt_j_post - gt_i_post))
    d_est = float(np.linalg.norm(est_j_post - est_i_post))
    change_gt, change_est = d_gt - d_pre, d_est - d_pre
    return {
        "preoperative_distance_mm": d_pre,
        "gt_postoperative_distance_mm": d_gt,
        "estimated_postoperative_distance_mm": d_est,
        "gt_distance_change_mm": change_gt,
        "estimated_distance_change_mm": change_est,
        "intervertebral_distance_change_error_mm": float(abs(change_est - change_gt)),
    }


def pair_metrics(gt_i: Iterable[float], gt_j: Iterable[float],
                 est_i: Iterable[float], est_j: Iterable[float],
                 centre_i_mm: Iterable[float], centre_j_mm: Iterable[float],
                 method: str = "independent_vertebrae") -> dict[str, float]:
    """Return both anatomy-aware metrics for one adjacent pair."""
    out = relative_pose_error(gt_i, gt_j, est_i, est_j, centre_i_mm, centre_j_mm, method)
    out.update(intervertebral_distance_error(gt_i, gt_j, est_i, est_j,
                                             centre_i_mm, centre_j_mm, method))
    return out

