"""Projective-geometry helpers for calibrated CT-to-radiograph checks.

The public API deliberately uses millimetres for 3-D points and pixels for
detector coordinates.  A projection matrix ``P`` is a 3x4 matrix that maps
homogeneous physical points ``[x_mm, y_mm, z_mm, 1]`` to homogeneous detector
coordinates ``[u, v, w]``.  Keeping this code independent of a DRR renderer
means that the same point-unit test can be used with synthetic C-arm matrices
and with a matrix read from PPCNet later.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def as_projection_matrix(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return a finite 3x4 float64 projection matrix."""
    p = np.asarray(matrix, dtype=np.float64)
    if p.shape != (3, 4):
        raise ValueError(f"projection matrix must have shape (3, 4), got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("projection matrix contains NaN or Inf")
    if np.linalg.matrix_rank(p) < 3:
        raise ValueError("projection matrix must have rank 3")
    return p


def project_points(points_mm: np.ndarray | list[list[float]], projection_matrix: np.ndarray) -> np.ndarray:
    """Project ``N x 3`` physical points to ``N x 2`` detector pixels.

    ``points_mm`` are expressed in the same physical coordinate frame used by
    the matrix.  Points whose homogeneous depth is zero are rejected because
    their pixel coordinates are undefined.  Negative depth is allowed here so
    that callers can decide whether a point is physically in front of a C-arm;
    :func:`projection_report` records it separately.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or Inf")
    p = as_projection_matrix(projection_matrix)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    image_h = (p @ homogeneous.T).T
    depth = image_h[:, 2]
    if np.any(np.isclose(depth, 0.0, atol=1e-12)):
        raise ValueError("at least one point projects to zero homogeneous depth")
    pixels = image_h[:, :2] / depth[:, None]
    return pixels.astype(np.float64, copy=False)


def reprojection_errors_px(
    predicted_px: np.ndarray | list[list[float]],
    observed_px: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Return Euclidean per-point detector errors in pixels."""
    predicted = np.asarray(predicted_px, dtype=np.float64)
    observed = np.asarray(observed_px, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted_px and observed_px must both have shape (N, 2)")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("pixel coordinates contain NaN or Inf")
    return np.linalg.norm(predicted - observed, axis=1)


def projection_report(
    points_mm: np.ndarray | list[list[float]],
    projection_matrix: np.ndarray,
    observed_px: np.ndarray | list[list[float]],
) -> dict[str, object]:
    """Create an auditable point-projection acceptance report.

    The acceptance criterion used by the project is both RMS and maximum
    reprojection error below one pixel.  The report also includes predicted
    pixels and a front-of-camera count for easier diagnosis.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    observed = np.asarray(observed_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("a projection acceptance report requires at least one correspondence")
    predicted = project_points(points, projection_matrix)
    errors = reprojection_errors_px(predicted, observed)
    p = as_projection_matrix(projection_matrix)
    homogeneous = np.c_[points, np.ones(len(points))]
    depth = (p @ homogeneous.T).T[:, 2]
    rms = float(np.sqrt(np.mean(errors**2)))
    maximum = float(np.max(errors))
    return {
        "n_points": int(len(points)),
        "predicted_pixels": predicted.tolist(),
        "observed_pixels": observed.tolist(),
        "errors_px": errors.tolist(),
        "rms_error_px": rms,
        "max_error_px": maximum,
        "front_of_camera_points": int(np.sum(depth > 0)),
        "all_points_in_front": bool(np.all(depth > 0)),
        "pass_lt_1px": bool(rms < 1.0 and maximum < 1.0),
    }


def load_projection_matrix(path: str | Path) -> np.ndarray:
    """Load a 3x4 matrix from ``.npy``, text, or JSON."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return as_projection_matrix(np.load(path))
    if path.suffix.lower() in {".txt", ".csv"}:
        return as_projection_matrix(np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("projection_matrix", "P", "projection", "matrix"):
            if key in payload:
                payload = payload[key]
                break
    return as_projection_matrix(payload)

