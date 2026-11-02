"""Case-level de-duplication fingerprints for external spine volumes.

PPCNet was curated from VerSe and CTSpine1K, so an external case cannot be
treated as independent only because its directory name differs from
``lumbar_XXXX``.  This module computes a conservative, auditable fingerprint
from the CT and label volumes.  It is deliberately independent of the
registration renderer and can be run before any synthetic target is created.

The fingerprint has three layers:

* exact byte hashes (useful when the same NIfTI was copied verbatim),
* canonical RAS, fixed-grid CT and segmentation digests (robust to NIfTI
  header/compression differences), and
* compact physical metadata and label-centroid summaries for near-match
  review.

Exact digests and near matches identify cases to exclude. A ``distinct``
fingerprint alone cannot establish patient independence after cropping or
resampling: an external cohort also needs a source-provenance audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


DEFAULT_GRID = (24, 24, 48)
DEFAULT_CT_MSE_THRESHOLD = 0.015
DEFAULT_SEG_MSE_THRESHOLD = 0.015


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _fixed_grid(values: np.ndarray, shape: tuple[int, int, int], *, order: int) -> np.ndarray:
    values = np.asarray(values)
    factors = tuple(float(dst) / float(src) for dst, src in zip(shape, values.shape))
    sampled = zoom(values, factors, order=order, mode="nearest", prefilter=False)
    # scipy can differ by one voxel due to rounding.  Crop/pad explicitly so
    # the digest is stable across scipy versions and source shapes.
    result = np.zeros(shape, dtype=sampled.dtype)
    slices_src = tuple(slice(0, min(int(a), int(b))) for a, b in zip(sampled.shape, shape))
    slices_dst = tuple(slice(0, min(int(a), int(b))) for a, b in zip(sampled.shape, shape))
    result[slices_dst] = sampled[slices_src]
    return result


def _normalised_ct_grid(data: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    # The clipping range covers lung/soft tissue and cortical bone while
    # reducing vendor-specific extreme values.  Per-volume normalisation is
    # intentionally avoided: absolute HU statistics remain part of metadata.
    clipped = np.clip(np.asarray(data, dtype=np.float32), -1000.0, 2000.0)
    quantised = np.rint((clipped + 1000.0) * (255.0 / 3000.0)).astype(np.uint8)
    return _fixed_grid(quantised, shape, order=1)


def _normalised_seg_grid(data: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    labels = np.rint(np.asarray(data, dtype=np.float32)).astype(np.int32)
    # A binary foreground digest is less sensitive to source label numbering;
    # the label inventory and centroids are retained separately for auditing.
    foreground = (labels > 0).astype(np.uint8)
    return _fixed_grid(foreground, shape, order=0)


def _header_strings(image: nib.spatialimages.SpatialImage) -> dict[str, str]:
    header = image.header
    result: dict[str, str] = {}
    for field in ("descrip", "aux_file", "db_name"):
        try:
            value = header[field].item()
        except Exception:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip("\x00 ")
        if text:
            result[field] = text
    return result


def _label_summary(labels: np.ndarray, affine: np.ndarray) -> list[dict[str, Any]]:
    values = np.rint(np.asarray(labels, dtype=np.float32)).astype(np.int32)
    summaries: list[dict[str, Any]] = []
    for source in sorted(int(v) for v in np.unique(values) if int(v) > 0):
        coords = np.argwhere(values == source)
        if coords.size == 0:
            continue
        centre = (np.asarray(affine, dtype=np.float64) @ np.r_[coords.mean(axis=0), 1.0])[:3]
        summaries.append({
            "label": source,
            "voxels": int(coords.shape[0]),
            "centroid_mm": np.round(centre, 3).astype(float).tolist(),
        })
    return summaries


def volume_fingerprint(ct_path: str | Path, seg_path: str | Path, *, grid: Iterable[int] = DEFAULT_GRID) -> dict[str, Any]:
    """Return a deterministic fingerprint and auditable volume metadata."""
    ct_path, seg_path = Path(ct_path).resolve(), Path(seg_path).resolve()
    target = tuple(int(v) for v in grid)
    if len(target) != 3 or any(v < 4 for v in target):
        raise ValueError(f"grid must contain three dimensions >=4, got {target}")
    ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
    seg_img = nib.as_closest_canonical(nib.load(str(seg_path)))
    if any(image.header.get_xyzt_units()[0] != "mm" for image in (ct_img, seg_img)):
        raise ValueError("CT/segmentation must declare millimetre spatial units before physical fingerprinting")
    ct = np.asarray(ct_img.get_fdata(dtype=np.float32))
    seg = np.asarray(seg_img.get_fdata(dtype=np.float32))
    if ct.ndim != 3 or seg.ndim != 3:
        raise ValueError(f"CT/segmentation must be 3-D: {ct.shape}, {seg.shape}")
    if ct.shape != seg.shape:
        # The registration preprocessor can trim zero label padding, but a
        # deduplication fingerprint must not hide a genuine grid mismatch.
        raise ValueError(f"CT/segmentation shape mismatch: {ct.shape} vs {seg.shape}")
    if not np.isfinite(ct).all() or not np.isfinite(seg).all():
        raise ValueError("CT/segmentation must contain only finite voxels")
    if np.any(seg < 0) or not np.allclose(seg, np.rint(seg), atol=1e-5, rtol=0) or seg.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("segmentation must contain nonnegative int32 labels")
    affine = np.asarray(ct_img.affine, dtype=np.float64)
    seg_affine = np.asarray(seg_img.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or not np.isfinite(seg_affine).all() or np.linalg.matrix_rank(affine) != 4 or np.linalg.matrix_rank(seg_affine) != 4:
        raise ValueError("CT/segmentation affines must be finite and invertible")
    if not np.allclose(affine, seg_affine, atol=1e-4, rtol=0):
        raise ValueError("CT/segmentation affines must describe the same physical grid")
    ct_grid = _normalised_ct_grid(ct, target)
    seg_grid = _normalised_seg_grid(seg, target)
    finite = ct.reshape(-1)
    spacing = np.asarray(ct_img.header.get_zooms()[:3], dtype=np.float64)
    foreground = seg > 0
    return {
        "ct_path": str(ct_path),
        "seg_path": str(seg_path),
        "ct_sha256": sha256_file(ct_path),
        "seg_sha256": sha256_file(seg_path),
        "canonical_shape": [int(v) for v in ct.shape],
        "spacing_mm": np.round(spacing, 6).astype(float).tolist(),
        "affine_mm": np.round(affine, 6).astype(float).tolist(),
        "ct_grid_shape": list(target),
        "ct_grid_sha256": _hash_array(ct_grid),
        "seg_grid_sha256": _hash_array(seg_grid),
        "ct_grid": ct_grid,
        "seg_grid": seg_grid,
        "ct_stats": {
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "p01": float(np.percentile(finite, 1.0)),
            "p50": float(np.percentile(finite, 50.0)),
            "p99": float(np.percentile(finite, 99.0)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        },
        "foreground_fraction": float(np.mean(foreground)),
        "label_summary": _label_summary(seg, affine),
        "ct_header": _header_strings(ct_img),
        "seg_header": _header_strings(seg_img),
    }


def _grid_mse(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32) / 255.0
    b = np.asarray(right, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def _seg_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Foreground overlap on the fixed grid; sparse-background MSE is misleading."""
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def compare_fingerprints(left: dict[str, Any], right: dict[str, Any], *, ct_mse_threshold: float = DEFAULT_CT_MSE_THRESHOLD, seg_mse_threshold: float = DEFAULT_SEG_MSE_THRESHOLD) -> dict[str, Any]:
    """Classify a pair as exact, near, or distinct.

    Near matches require both intensity and foreground geometry agreement. A
    single exact/canonical modality match is already enough to block a case.
    ``distinct`` means this screen did not detect a match; it is not evidence
    of patient independence. The legacy seg_mse_threshold is diagnostic only;
    near matches use foreground IoU to avoid sparse-background bias.
    """
    for fingerprint in (left, right):
        for key in ("ct_sha256", "seg_sha256", "ct_grid_sha256", "seg_grid_sha256"):
            value = fingerprint.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"fingerprint requires a valid SHA256 field: {key}")
        ct_grid, seg_grid = np.asarray(fingerprint["ct_grid"]), np.asarray(fingerprint["seg_grid"])
        if ct_grid.ndim != 3 or min(ct_grid.shape) < 1 or seg_grid.shape != ct_grid.shape:
            raise ValueError("fingerprint grids must be nonempty matching 3-D arrays")
        if not np.isfinite(ct_grid).all() or np.any(ct_grid < 0) or np.any(ct_grid > 255) or not np.isin(seg_grid, [0, 1]).all():
            raise ValueError("fingerprint CT grid must be finite [0,255] and segmentation grid binary")
        if _hash_array(ct_grid) != fingerprint["ct_grid_sha256"] or _hash_array(seg_grid) != fingerprint["seg_grid_sha256"]:
            raise ValueError("fingerprint grid digest does not match its array")
    if np.shape(left["ct_grid"]) != np.shape(right["ct_grid"]):
        raise ValueError("fingerprints must use the same sampling grid")
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in (ct_mse_threshold, seg_mse_threshold)):
        raise ValueError("fingerprint MSE thresholds must be finite in [0,1]")
    exact_ct = left.get("ct_sha256") == right.get("ct_sha256")
    exact_seg = left.get("seg_sha256") == right.get("seg_sha256")
    canonical_ct = left.get("ct_grid_sha256") == right.get("ct_grid_sha256")
    canonical_seg = left.get("seg_grid_sha256") == right.get("seg_grid_sha256")
    ct_mse = _grid_mse(left["ct_grid"], right["ct_grid"])
    seg_mse = float(np.mean((np.asarray(left["seg_grid"], dtype=np.float32) - np.asarray(right["seg_grid"], dtype=np.float32)) ** 2))
    seg_iou = _seg_iou(left["seg_grid"], right["seg_grid"])
    # Segmentation MSE is dominated by the empty background and is therefore
    # near zero for almost every unrelated spine. Require foreground overlap
    # together with CT similarity for a near-duplicate decision.
    near = (ct_mse <= float(ct_mse_threshold) and seg_iou >= 0.75)
    # Either modality can identify a duplicate.  In particular, the same
    # segmentation paired with a re-windowed/recompressed CT must not slip
    # through as an "independent" patient.
    if exact_ct or exact_seg:
        status = "exact_duplicate"
    elif canonical_ct or canonical_seg:
        status = "canonical_duplicate"
    elif near:
        status = "near_duplicate"
    else:
        status = "distinct"
    return {
        "status": status,
        "exact_ct": bool(exact_ct),
        "exact_seg": bool(exact_seg),
        "canonical_ct": bool(canonical_ct),
        "canonical_seg": bool(canonical_seg),
        "ct_mse": ct_mse,
        "seg_mse": seg_mse,
        "seg_iou": seg_iou,
        "ct_mse_threshold": float(ct_mse_threshold),
        "seg_mse_threshold": float(seg_mse_threshold),
        "seg_mse_threshold_usage": "diagnostic_only; near-duplicate decision uses foreground IoU >= 0.75",
        "independence_established": False,
        "decision_note": "screening only; source provenance is required to establish independent patients",
    }


def strip_grid(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Remove large arrays before serialising a manifest."""
    return {key: value for key, value in fingerprint.items() if key not in {"ct_grid", "seg_grid"}}


def json_dump(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

