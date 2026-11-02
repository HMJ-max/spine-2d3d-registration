"""Small SE(3) pose helpers used by tests and future registration code."""
from __future__ import annotations
import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("rigid pose must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0):
        raise ValueError("rotation must be a proper orthonormal matrix")
    out = np.eye(4, dtype=float)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("transform must be a homogeneous 4x4 rigid matrix")
    make_transform(transform[:3, :3], transform[:3, 3])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N x 3 array")
    hom = np.c_[points, np.ones(len(points))]
    return (transform @ hom.T).T[:, :3]
