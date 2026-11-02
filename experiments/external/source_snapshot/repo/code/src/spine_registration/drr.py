"""DRR renderers and global rigid registration baselines.

The renderer is intentionally a transparent research baseline rather than a
clinical X-ray simulator.  It projects a CT/segmentation volume by summing
attenuation along one axis after a voxel-space rigid transform.  This is
enough to validate data plumbing, pose conventions, losses and optimization
The module contains both the original orthographic development renderer and a
calibrated, physical-mm cone-beam renderer.  The latter is deliberately small
and transparent: it ray-integrates a volume through a pinhole C-arm model and
accepts a native 3x4 projection matrix so that PPCNet matrices can be used
without changing the downstream code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .pose import make_transform
# Re-export the calibrated point projector alongside the renderer API.  This
# keeps downstream code compatible with PPCNet's native 3x4 matrices while
# the numerical implementation remains in the small geometry helper module.
from .projection import project_points


@dataclass
class Volume:
    data: np.ndarray
    spacing_mm: np.ndarray
    path: str | None = None
    affine_mm: np.ndarray | None = None

    def voxel_to_world_affine(self) -> np.ndarray:
        """Return a voxel-index to physical-mm affine.

        NIfTI inputs keep their native affine.  Resampled NPZ/NPY inputs use a
        centred axis-aligned frame so that their physical coordinates remain
        auditable and compatible with the virtual C-arm contract.
        """
        if self.affine_mm is not None:
            affine = np.asarray(self.affine_mm, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(affine).all():
                raise ValueError("affine_mm contains NaN or Inf")
            if not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
                raise ValueError("affine_mm must be a nonsingular homogeneous affine")
            return affine
        spacing = np.asarray(self.spacing_mm, dtype=np.float64).reshape(3)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing_mm must be finite and positive")
        lower = -0.5 * (np.asarray(self.data.shape, dtype=np.float64) - 1.0) * spacing
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = np.diag(spacing)
        affine[:3, 3] = lower
        return affine


@dataclass
class CArmGeometry:
    """Pinhole C-arm geometry expressed in physical millimetres.

    The world origin is the isocentre.  The AP view places the source at
    ``(0, 0, -source_to_isocenter_mm)`` and looks along +z.  ``view_angle_deg``
    rotates the C-arm around the world y axis, so 90 degrees is a lateral
    view.  Detector coordinates are ``(u, v)`` pixels and images are returned
    as ``(height, width)`` arrays.
    """

    detector_size_px: tuple[int, int] = (256, 256)
    pixel_spacing_mm: tuple[float, float] = (0.6, 0.6)
    source_to_isocenter_mm: float = 750.0
    source_to_detector_mm: float = 1200.0
    principal_point_px: tuple[float, float] | None = None
    view_angle_deg: float = 0.0
    rotation_axis: str = "y"

    def __post_init__(self) -> None:
        h, w = (int(self.detector_size_px[0]), int(self.detector_size_px[1]))
        sx, sy = (float(self.pixel_spacing_mm[0]), float(self.pixel_spacing_mm[1]))
        if h < 2 or w < 2:
            raise ValueError("detector_size_px must be at least 2x2")
        if sx <= 0 or sy <= 0:
            raise ValueError("pixel_spacing_mm must be positive")
        if self.source_to_isocenter_mm <= 0 or self.source_to_detector_mm <= 0:
            raise ValueError("source and detector distances must be positive")
        if self.source_to_detector_mm <= self.source_to_isocenter_mm:
            raise ValueError("source_to_detector_mm must exceed source_to_isocenter_mm")
        if self.principal_point_px is None:
            self.principal_point_px = ((w - 1) / 2.0, (h - 1) / 2.0)
        else:
            self.principal_point_px = (float(self.principal_point_px[0]), float(self.principal_point_px[1]))
        self.detector_size_px = (h, w)
        self.pixel_spacing_mm = (sx, sy)
        self.rotation_axis = str(self.rotation_axis).lower()
        if self.rotation_axis not in {"x", "y", "z"}:
            raise ValueError("rotation_axis must be x, y or z")

    @property
    def focal_length_px(self) -> tuple[float, float]:
        return (
            float(self.source_to_detector_mm / self.pixel_spacing_mm[0]),
            float(self.source_to_detector_mm / self.pixel_spacing_mm[1]),
        )

    def camera_center_mm(self) -> np.ndarray:
        """Return the X-ray source position in the world frame."""
        r_wc = Rotation.from_euler(self.rotation_axis, self.view_angle_deg, degrees=True).as_matrix()
        return r_wc @ np.array([0.0, 0.0, -float(self.source_to_isocenter_mm)])

    def projection_matrix(self) -> np.ndarray:
        """Return a 3x4 matrix mapping world-mm points to detector pixels."""
        r_wc = Rotation.from_euler(self.rotation_axis, self.view_angle_deg, degrees=True).as_matrix()
        r_cw = r_wc.T
        center = self.camera_center_mm()
        t = -r_cw @ center
        fx, fy = self.focal_length_px
        u0, v0 = self.principal_point_px  # type: ignore[misc]
        k = np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return (k @ np.column_stack([r_cw, t])).astype(np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "detector_size_px": list(self.detector_size_px),
            "pixel_spacing_mm": list(self.pixel_spacing_mm),
            "source_to_isocenter_mm": float(self.source_to_isocenter_mm),
            "source_to_detector_mm": float(self.source_to_detector_mm),
            "principal_point_px": list(self.principal_point_px),
            "view_angle_deg": float(self.view_angle_deg),
            "rotation_axis": self.rotation_axis,
            "focal_length_px": list(self.focal_length_px),
            "projection_matrix": self.projection_matrix().tolist(),
            "world_frame": "volume-centred physical millimetres",
        }


def make_projection_matrix(geometry: CArmGeometry) -> np.ndarray:
    """Compatibility helper returning the calibrated 3x4 matrix for a C-arm."""
    if not isinstance(geometry, CArmGeometry):
        raise TypeError("geometry must be a CArmGeometry instance")
    return geometry.projection_matrix()


def load_projection_matrix(path: str | Path) -> np.ndarray:
    """Load a PPCNet-compatible 3x4 matrix from NPY, JSON or TXT."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() in {".txt", ".csv"}:
        matrix = np.loadtxt(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("projection_matrix", "P", "projection", "matrix"):
                if key in payload:
                    payload = payload[key]
                    break
        matrix = payload
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"projection matrix must be a finite rank-3 array with shape (3, 4), got {matrix.shape}")
    return matrix


def load_volume(path: str | Path, key: str = "volume") -> Volume:
    """Load NIfTI, NPY, or NPZ volume and return a float32 array.

    For NPZ files, ``key`` is preferred and ``volume``/``image``/``ct`` are
    tried as fallbacks.  NIfTI spacing is read from the header; NPY/NPZ data
    defaults to 1 mm isotropic spacing unless a ``spacing_mm`` array exists.
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    spacing = np.ones(3, dtype=np.float32)
    affine_mm: np.ndarray | None = None
    if suffixes.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("NIfTI input requires nibabel") from exc
        obj = nib.load(str(path))
        data = np.asarray(obj.get_fdata(dtype=np.float32))
        spacing = np.asarray(obj.header.get_zooms()[:3], dtype=np.float32)
        affine_mm = np.asarray(obj.affine, dtype=np.float64)
    elif path.suffix.lower() == ".npy":
        data = np.asarray(np.load(path), dtype=np.float32)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        candidates = [key, "volume", "image", "ct", "data", "segmentation", "mask"]
        name = next((k for k in candidates if k in archive), None)
        if name is None:
            raise KeyError(f"No volume-like array found in {path}; keys={list(archive.keys())}")
        data = np.asarray(archive[name], dtype=np.float32)
        if "spacing_mm" in archive:
            spacing = np.asarray(archive["spacing_mm"], dtype=np.float32).reshape(3)
        if "affine_mm" in archive:
            affine_mm = np.asarray(archive["affine_mm"], dtype=np.float64).reshape(4, 4)
    else:
        raise ValueError(f"Unsupported volume format: {path}")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return Volume(data=data, spacing_mm=spacing, path=str(path), affine_mm=affine_mm)


def _normalise(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.5]) if image.size else (0.0, 1.0)
    if hi <= lo:
        lo, hi = float(image.min(initial=0.0)), float(image.max(initial=1.0))
    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return image.astype(np.float32, copy=False)


def _apply_transform(volume: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Resample the source volume after a voxel-space transform.

    ``transform`` maps source voxel coordinates to output voxel coordinates.
    scipy's affine_transform samples output coordinates from the input, hence
    the inverse matrix and inverse translation are used here.
    """
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inv = np.linalg.inv(transform)
    return ndimage.affine_transform(
        volume,
        matrix=inv[:3, :3],
        offset=inv[:3, 3],
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def _view_volume(volume: np.ndarray, view: str, angle_deg: float = 0.0) -> np.ndarray:
    """Orient a volume for the requested projection view.

    AP sums the last (z) axis, LAT sums the first (x) axis. OBL rotates the
    volume about its y axis before summing z. The exact C-arm geometry is
    deliberately represented by the angle in metadata and can later be
    replaced by a calibrated projection matrix.
    """
    view = view.upper()
    if view == "LAT":
        oriented = np.transpose(volume, (2, 1, 0))
        if angle_deg:
            oriented = ndimage.rotate(oriented, angle_deg, axes=(0, 2), reshape=False, order=1, mode="constant")
        return oriented
    if view == "OBL" or angle_deg:
        return ndimage.rotate(volume, angle_deg, axes=(0, 2), reshape=False, order=1, mode="constant")
    if view != "AP":
        raise ValueError(f"view must be AP, LAT or OBL, got {view}")
    return volume


def render_drr(
    volume: np.ndarray | Volume,
    transform: np.ndarray | None = None,
    view: str = "AP",
    angle_deg: float = 0.0,
    output_size: tuple[int, int] | None = None,
    log_transform: bool = True,
) -> np.ndarray:
    """Render a normalized orthographic DRR as a float32 image in [0, 1]."""
    array = volume.data if isinstance(volume, Volume) else np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("render_drr expects a 3-D array")
    moved = _apply_transform(array, transform) if transform is not None else array
    oriented = _view_volume(moved, view=view, angle_deg=angle_deg)
    # Summing attenuation is a simple Beer-Lambert proxy. For masks, max is
    # less destructive; the sum retains more shape information for CT data.
    projection = oriented.sum(axis=0)
    if log_transform:
        projection = np.log1p(np.maximum(projection, 0.0))
    image = _normalise(projection)
    if output_size is not None and tuple(image.shape) != tuple(output_size):
        zoom = (output_size[0] / image.shape[0], output_size[1] / image.shape[1])
        image = ndimage.zoom(image, zoom, order=1)
    return image.astype(np.float32, copy=False)


def params_to_transform(params: Iterable[float]) -> np.ndarray:
    """Convert [rx, ry, rz, tx, ty, tz] to a 4x4 voxel-space SE(3) matrix."""
    values = np.asarray(list(params), dtype=np.float64).reshape(6)
    rotation = Rotation.from_rotvec(values[:3]).as_matrix()
    return make_transform(rotation, values[3:])


def transform_to_params(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def ncc_loss(reference: np.ndarray, moving: np.ndarray) -> float:
    """One minus zero-mean normalized cross correlation."""
    if np.shape(reference) != np.shape(moving) or np.size(reference) == 0:
        raise ValueError("NCC images must have the same nonempty shape")
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(moving, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 1.0
    return float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 2.0))


def register_global_rigid(
    volume: np.ndarray | Volume,
    target: np.ndarray,
    view: str = "AP",
    angle_deg: float = 0.0,
    initial_params: Iterable[float] | None = None,
    max_iterations: int = 80,
    output_size: tuple[int, int] | None = None,
    translation_bound: float | None = None,
) -> dict:
    """Optimize one global six-parameter rigid transform against a DRR.

    Rotation parameters are axis-angle radians and translations are voxels.
    Powell is used because this baseline must also run without a differentiable
    renderer. The returned dictionary is JSON serializable except for the
    optional ``transform`` array, which is converted to a nested list.
    """
    target = _normalise(np.asarray(target, dtype=np.float32))
    if output_size is None:
        output_size = tuple(target.shape)
    initial = np.zeros(6, dtype=np.float64) if initial_params is None else np.asarray(list(initial_params), dtype=np.float64)
    if initial.shape != (6,):
        raise ValueError("initial_params must contain [rx, ry, rz, tx, ty, tz]")
    history: list[float] = []

    def objective(params: np.ndarray) -> float:
        prediction = render_drr(volume, params_to_transform(params), view, angle_deg, output_size)
        loss = ncc_loss(target, prediction)
        history.append(loss)
        return loss

    bounds = None
    if translation_bound is not None:
        bounds = [(-np.pi, np.pi)] * 3 + [(-translation_bound, translation_bound)] * 3
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(max_iterations), "xtol": 1e-4, "ftol": 1e-5, "disp": False},
    )
    estimated = params_to_transform(result.x)
    prediction = render_drr(volume, estimated, view, angle_deg, output_size)
    return {
        "params": result.x.tolist(),
        "transform": estimated.tolist(),
        "initial_params": initial.tolist(),
        "final_loss": float(ncc_loss(target, prediction)),
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "n_evaluations": int(getattr(result, "nfev", len(history))),
        "loss_history": [float(v) for v in history],
        "prediction": prediction,
    }


def pose_error(estimated: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Translation L2 error and geodesic rotation error in degrees."""
    estimated = np.asarray(estimated, dtype=np.float64).reshape(4, 4)
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(4, 4)
    delta = np.linalg.inv(ground_truth) @ estimated
    angle = np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
    return {
        "translation_error_vox": float(np.linalg.norm(delta[:3, 3])),
        "rotation_error_deg": float(np.degrees(angle)),
    }


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), np.asarray(image, dtype=np.float32))
    try:
        from imageio.v3 import imwrite
        imwrite(path.with_suffix(".png"), (np.clip(image, 0, 1) * 255).astype(np.uint8))
    except Exception:
        # NPY is the canonical output; PNG is a convenience for inspection.
        pass


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def register_global_rigid_multiview(
    volume: np.ndarray | Volume,
    targets: Iterable[np.ndarray],
    views: Iterable[str],
    angles_deg: Iterable[float] | None = None,
    initial_params: Iterable[float] | None = None,
    max_iterations: int = 80,
    translation_bound: float | None = None,
) -> dict:
    """Joint global-rigid optimization over one or more calibrated views.

    With AP alone the translation along the ray is inherently weakly
    observable. Supplying AP+LAT (and optionally OBL) removes that ambiguity
    in this toy renderer and mirrors the biplanar experiment protocol.
    """
    target_list = [_normalise(np.asarray(t, dtype=np.float32)) for t in targets]
    view_list = [str(v).upper() for v in views]
    if len(target_list) != len(view_list):
        raise ValueError("targets and views must have the same length")
    if angles_deg is None:
        angle_list = [0.0] * len(target_list)
    else:
        angle_list = [float(v) for v in angles_deg]
        if len(angle_list) != len(target_list):
            raise ValueError("angles_deg must have one value per target")
    initial = np.zeros(6, dtype=np.float64) if initial_params is None else np.asarray(list(initial_params), dtype=np.float64).reshape(6)
    history: list[float] = []

    def objective(params: np.ndarray) -> float:
        losses = []
        transform = params_to_transform(params)
        for target, view, angle in zip(target_list, view_list, angle_list):
            pred = render_drr(volume, transform, view, angle, tuple(target.shape))
            losses.append(ncc_loss(target, pred))
        value = float(np.mean(losses))
        history.append(value)
        return value

    bounds = None
    if translation_bound is not None:
        bounds = [(-np.pi, np.pi)] * 3 + [(-translation_bound, translation_bound)] * 3
    result = minimize(objective, initial, method="Powell", bounds=bounds,
                      options={"maxiter": int(max_iterations), "xtol": 1e-4, "ftol": 1e-5, "disp": False})
    transform = params_to_transform(result.x)
    predictions = [render_drr(volume, transform, view, angle, tuple(target.shape))
                   for target, view, angle in zip(target_list, view_list, angle_list)]
    return {
        "params": result.x.tolist(),
        "transform": transform.tolist(),
        "initial_params": initial.tolist(),
        "final_loss": float(np.mean([ncc_loss(t, p) for t, p in zip(target_list, predictions)])),
        "per_view_loss": [float(ncc_loss(t, p)) for t, p in zip(target_list, predictions)],
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "n_evaluations": int(getattr(result, "nfev", len(history))),
        "loss_history": [float(v) for v in history],
        "predictions": predictions,
    }


def _camera_center_from_projection_matrix(projection_matrix: np.ndarray) -> np.ndarray:
    """Return the world-mm camera centre (right null vector of P)."""
    p = np.asarray(projection_matrix, dtype=np.float64).reshape(3, 4)
    _, _, vh = np.linalg.svd(p)
    center_h = vh[-1]
    if abs(center_h[3]) < 1e-12:
        raise ValueError("projection matrix has a camera centre at infinity")
    return center_h[:3] / center_h[3]


def _volume_bounds_mm(volume: Volume, origin_mm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return axis-aligned world-mm bounds for a volume."""
    if origin_mm is not None:
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = np.diag(np.asarray(volume.spacing_mm, dtype=np.float64).reshape(3))
        affine[:3, 3] = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    else:
        affine = volume.voxel_to_world_affine()
    shape = np.asarray(volume.data.shape, dtype=np.float64) - 1.0
    corners = np.array(
        [[i, j, k, 1.0] for i in (0.0, shape[0])
         for j in (0.0, shape[1])
         for k in (0.0, shape[2])], dtype=np.float64
    )
    world = (affine @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _ray_box_intersections(
    source_mm: np.ndarray,
    directions: np.ndarray,
    lower_mm: np.ndarray,
    upper_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect N rays with an axis-aligned volume box."""
    source = np.asarray(source_mm, dtype=np.float64).reshape(1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    parallel = np.abs(directions) < 1e-12
    divisor = np.where(parallel, 1.0, directions)
    t0 = (lower_mm.reshape(1, 3) - source) / divisor
    t1 = (upper_mm.reshape(1, 3) - source) / divisor
    outside = np.any(parallel & ((source < lower_mm) | (source > upper_mm)), axis=1)
    t0 = np.where(parallel, -np.inf, t0)
    t1 = np.where(parallel, np.inf, t1)
    near = np.max(np.minimum(t0, t1), axis=1)
    far = np.min(np.maximum(t0, t1), axis=1)
    valid = ~outside & np.isfinite(near) & np.isfinite(far) & (far > np.maximum(near, 0.0))
    near = np.maximum(near, 0.0)
    return np.where(valid, near, 0.0), np.where(valid, far, 0.0), valid


def _integration_sample_count(volume: Volume, voxel_to_world: np.ndarray,
                              sample_step_mm: float | None, max_samples: int) -> int:
    """Pose- and chunk-independent midpoint quadrature count for rigid motion."""
    step = float(sample_step_mm) if sample_step_mm is not None else max(float(np.min(volume.spacing_mm)) * 0.75, 0.25)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("sample_step_mm must be finite and positive")
    extent = np.asarray(volume.data.shape, dtype=np.float64) - 1
    # Largest physical corner-to-corner diagonal also handles oblique affines.
    diagonals = np.array([[i, j, k] for i in (-extent[0], extent[0])
                          for j in (-extent[1], extent[1]) for k in (-extent[2], extent[2])])
    diameter = np.linalg.norm(diagonals @ voxel_to_world[:3, :3].T, axis=1).max()
    return int(np.clip(np.ceil(diameter / step), 2, max_samples))


def _rays_from_projection_matrix(projection_matrix: np.ndarray, pixels_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build source-to-detector rays for arbitrary full-rank 3x4 P."""
    p = np.asarray(projection_matrix, dtype=np.float64).reshape(3, 4)
    m = p[:, :3]
    if abs(np.linalg.det(m)) < 1e-12:
        raise ValueError("projection matrix left 3x3 block is singular")
    source = _camera_center_from_projection_matrix(p)
    homogeneous_pixels = np.c_[np.asarray(pixels_uv, dtype=np.float64), np.ones(len(pixels_uv))]
    directions = (np.linalg.inv(m) @ homogeneous_pixels.T).T
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    return source, directions


def render_cone_beam_drr(
    volume: np.ndarray | Volume,
    geometry: CArmGeometry | None = None,
    projection_matrix: np.ndarray | None = None,
    transform_mm: np.ndarray | None = None,
    output_size: tuple[int, int] | None = None,
    origin_mm: np.ndarray | None = None,
    sample_step_mm: float | None = None,
    max_samples: int = 256,
    chunk_rows: int = 16,
    log_transform: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    """Render a calibrated cone-beam DRR using physical millimetres.

    ``projection_matrix`` maps homogeneous world-mm points to detector pixels.
    If omitted, it is generated from ``geometry``.  The volume is interpreted
    in a centred physical frame unless ``origin_mm`` is supplied.  ``transform_mm``
    maps source volume coordinates to world coordinates and therefore uses
    millimetre translations.  Ray integration is chunked by detector rows to
    keep a 512x512 image practical on a laptop GPU/CPU.
    """
    vol = volume if isinstance(volume, Volume) else Volume(np.asarray(volume, dtype=np.float32), np.ones(3, dtype=np.float32))
    if vol.data.ndim != 3 or min(vol.data.shape) < 2:
        raise ValueError("render_cone_beam_drr expects a 3-D volume with at least two samples per axis")
    if projection_matrix is None:
        if geometry is None:
            geometry = CArmGeometry()
        if output_size is not None and tuple(output_size) != tuple(geometry.detector_size_px):
            geometry = CArmGeometry(
                detector_size_px=tuple(map(int, output_size)),
                pixel_spacing_mm=geometry.pixel_spacing_mm,
                source_to_isocenter_mm=geometry.source_to_isocenter_mm,
                source_to_detector_mm=geometry.source_to_detector_mm,
                principal_point_px=None,
                view_angle_deg=geometry.view_angle_deg,
                rotation_axis=geometry.rotation_axis,
            )
        p = geometry.projection_matrix()
        output_size = geometry.detector_size_px
    else:
        p = np.asarray(projection_matrix, dtype=np.float64).reshape(3, 4)
        if p.shape != (3, 4) or not np.isfinite(p).all() or np.linalg.matrix_rank(p) < 3:
            raise ValueError("projection_matrix must be a finite rank-3 (3, 4) matrix")
        if output_size is None:
            if geometry is None:
                raise ValueError("output_size or geometry is required when passing a projection_matrix")
            output_size = geometry.detector_size_px
    height, width = (int(output_size[0]), int(output_size[1]))
    if height < 2 or width < 2:
        raise ValueError("output_size must be at least 2x2")
    if chunk_rows < 1 or max_samples < 2:
        raise ValueError("chunk_rows must be positive and max_samples must be at least 2")

    source, _ = _rays_from_projection_matrix(p, np.array([[0.0, 0.0]], dtype=np.float64))
    data = np.asarray(vol.data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("volume contains NaN or Inf")
    if origin_mm is None:
        voxel_to_world = vol.voxel_to_world_affine()
    else:
        voxel_to_world = np.eye(4, dtype=np.float64)
        voxel_to_world[:3, :3] = np.diag(np.asarray(vol.spacing_mm, dtype=np.float64).reshape(3))
        voxel_to_world[:3, 3] = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    world_transform = np.eye(4, dtype=np.float64) if transform_mm is None else np.asarray(transform_mm, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(world_transform).all():
        raise ValueError("transform_mm contains NaN or Inf")
    world_to_local = np.linalg.inv(voxel_to_world) @ np.linalg.inv(world_transform)
    local_source = world_to_local[:3, :3] @ source + world_to_local[:3, 3]
    lower, upper = np.zeros(3), np.asarray(vol.data.shape, dtype=np.float64) - 1
    n_samples = _integration_sample_count(vol, voxel_to_world, sample_step_mm, max_samples)
    fractions = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    image = np.zeros((height, width), dtype=np.float32)
    u_grid = np.arange(width, dtype=np.float64)
    v_grid = np.arange(height, dtype=np.float64)
    for row_start in range(0, height, int(chunk_rows)):
        row_stop = min(height, row_start + int(chunk_rows))
        uu, vv = np.meshgrid(u_grid, v_grid[row_start:row_stop], indexing="xy")
        pixels = np.c_[uu.ravel(), vv.ravel()]
        _, directions = _rays_from_projection_matrix(p, pixels)
        local_directions = directions @ world_to_local[:3, :3].T
        near, far, valid = _ray_box_intersections(local_source, local_directions, lower, upper)
        lengths = far - near
        distances = near[:, None] + lengths[:, None] * fractions[None, :]
        indices = local_source[None, None, :] + distances[..., None] * local_directions[:, None, :]
        samples = ndimage.map_coordinates(data, indices.reshape(-1, 3).T, order=1,
                                           mode="grid-constant", cval=0.0, prefilter=False).reshape(len(pixels), n_samples)
        samples[~valid, :] = 0.0
        line_integral = samples.mean(axis=1) * lengths
        if log_transform:
            line_integral = np.log1p(np.maximum(line_integral, 0.0))
        image[row_start:row_stop] = line_integral.reshape(row_stop - row_start, width).astype(np.float32)
    # Keep the historical normalized output by default.  Synthetic motion
    # experiments also need the additive line-integral image before display
    # normalization, hence the explicit opt-out used by the experiment CLI.
    return _normalise(image) if normalize_output else image.astype(np.float32, copy=False)


def register_global_rigid_cone_multiview(
    volume: np.ndarray | Volume,
    targets: Iterable[np.ndarray],
    projection_matrices: Iterable[np.ndarray],
    initial_params_mm: Iterable[float] | None = None,
    max_iterations: int = 20,
    translation_bound_mm: float | None = None,
    sample_step_mm: float | None = None,
    max_samples: int = 256,
    chunk_rows: int = 16,
) -> dict:
    """Global SE(3) baseline for calibrated cone-beam DRRs.

    The six parameters are ``[rx, ry, rz, tx_mm, ty_mm, tz_mm]``.  Each
    target owns its own native 3x4 projection matrix, so AP/LAT/OBL views can
    be mixed without converting them into a common image geometry.
    """
    target_list = [_normalise(np.asarray(t, dtype=np.float32)) for t in targets]
    p_list = [np.asarray(p, dtype=np.float64).reshape(3, 4) for p in projection_matrices]
    if len(target_list) != len(p_list) or not target_list:
        raise ValueError("targets and projection_matrices must have the same non-zero length")
    initial = np.zeros(6, dtype=np.float64) if initial_params_mm is None else np.asarray(list(initial_params_mm), dtype=np.float64).reshape(6)
    history: list[float] = []

    def objective(params: np.ndarray) -> float:
        transform_mm = params_to_transform(params)
        losses = []
        for target, p in zip(target_list, p_list):
            pred = render_cone_beam_drr(
                volume,
                projection_matrix=p,
                output_size=tuple(target.shape),
                transform_mm=transform_mm,
                sample_step_mm=sample_step_mm,
                max_samples=max_samples,
                chunk_rows=chunk_rows,
            )
            losses.append(ncc_loss(target, pred))
        value = float(np.mean(losses))
        history.append(value)
        return value

    bounds = None
    if translation_bound_mm is not None:
        bounds = [(-np.pi, np.pi)] * 3 + [(-float(translation_bound_mm), float(translation_bound_mm))] * 3
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(max_iterations), "xtol": 1e-4, "ftol": 1e-5, "disp": False},
    )
    transform_mm = params_to_transform(result.x)
    predictions = [
        render_cone_beam_drr(
            volume,
            projection_matrix=p,
            output_size=tuple(target.shape),
            transform_mm=transform_mm,
            sample_step_mm=sample_step_mm,
            max_samples=max_samples,
            chunk_rows=chunk_rows,
        )
        for target, p in zip(target_list, p_list)
    ]
    return {
        "params_rx_ry_rz_tx_ty_tz_mm": result.x.tolist(),
        "transform_mm": transform_mm.tolist(),
        "initial_params_mm": initial.tolist(),
        "final_loss": float(np.mean([ncc_loss(t, p) for t, p in zip(target_list, predictions)])),
        "per_view_loss": [float(ncc_loss(t, p)) for t, p in zip(target_list, predictions)],
        "success": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "n_evaluations": int(getattr(result, "nfev", len(history))),
        "loss_history": [float(v) for v in history],
        "predictions": predictions,
        "coordinate_units": "physical_mm",
    }
