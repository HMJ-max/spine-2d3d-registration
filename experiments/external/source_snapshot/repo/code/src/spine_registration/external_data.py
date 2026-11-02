"""Utilities for adapting external vertebral segmentations to L1--L5.

The public datasets used by the project do not share one segmentation naming
convention.  This module deliberately keeps the adapter independent of the
registration code: it discovers NIfTI image/label pairs, computes physical
centroids from the NIfTI affine, and returns a deterministic source-label to
``L1``--``L5`` mapping.  Standard VerSe/CTSpine1K label ids (20--24) are
preferred when present. Other conventions require an explicit, anatomically
verified label mapping; spatial ordering alone cannot identify L1--L5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import nibabel as nib
import numpy as np


_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_IMAGE_TOKENS = ("ct", "image", "img", "scan", "volume")
_LABEL_TOKENS = ("seg", "label", "labels", "mask", "vertebra", "vertebrae", "annotation", "ann")


@dataclass(frozen=True)
class ExternalCase:
    """A discovered CT/segmentation pair."""

    dataset: str
    case_id: str
    ct: str
    seg: str
    source_dir: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _stem(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem.lower()


def _tokens(stem: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", stem) if token}


def _kind(path: Path) -> str | None:
    tokens = _tokens(_stem(path))
    if tokens & set(_LABEL_TOKENS):
        return "seg"
    if tokens & set(_IMAGE_TOKENS):
        return "ct"
    # Dataset-specific fallbacks.  Files named ``case_001``/``case_001_gt``
    # should not be silently treated as CT unless a label partner exists.
    if any(token in _stem(path) for token in ("label", "seg", "mask", "annot")):
        return "seg"
    return None


def discover_external_cases(root: str | Path, *, dataset_names: Iterable[str] | None = None) -> tuple[list[ExternalCase], list[dict[str, Any]]]:
    """Discover CT/segmentation NIfTI pairs below ``root``.

    Pairing is intentionally conservative: image and label files must reside
    in the same directory (or one immediate child directory) and a directory
    contributes at most one pair.  Ambiguous pairs are returned in the error
    list instead of being guessed.
    """
    root = Path(root).expanduser().resolve()
    wanted = {str(x).lower() for x in dataset_names} if dataset_names else None
    errors: list[dict[str, Any]] = []
    cases: list[ExternalCase] = []
    if not root.exists():
        return [], [{"root": str(root), "error": "root_missing"}]
    for directory in sorted({p.parent for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(_NIFTI_SUFFIXES)}):
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.name.lower().endswith(_NIFTI_SUFFIXES))
        ct_files = [p for p in files if _kind(p) == "ct"]
        seg_files = [p for p in files if _kind(p) == "seg"]
        # If the directory has no explicit CT token, use one unclassified
        # image only when a segmentation partner is unambiguous.
        if not ct_files:
            unknown = [p for p in files if _kind(p) is None]
            if len(unknown) == 1:
                ct_files = unknown
        if len(ct_files) != 1 or len(seg_files) != 1:
            if ct_files or seg_files:
                errors.append({"directory": str(directory), "ct_candidates": [str(p) for p in ct_files], "seg_candidates": [str(p) for p in seg_files], "error": "ambiguous_or_incomplete_pair"})
            continue
        # Prefer the nearest semantic dataset directory; callers may filter
        # using --dataset, while this default remains useful for nested roots.
        relative_parts = directory.relative_to(root).parts
        # A dataset root may itself contain case directories (PPCNet-style),
        # while external roots usually contain ``VerSe/<case>`` and
        # ``CTSpine1K/<case>``.  Treat direct children as belonging to the
        # root dataset so ``--dataset PPCNet`` remains useful for smoke runs.
        dataset = relative_parts[0] if len(relative_parts) >= 2 else root.name
        if wanted and dataset.lower() not in wanted:
            continue
        case_id = directory.name
        cases.append(ExternalCase(dataset=dataset, case_id=case_id, ct=str(ct_files[0].resolve()), seg=str(seg_files[0].resolve()), source_dir=str(directory.resolve())))
    # Deduplicate accidental nested discoveries and make ordering stable.
    unique: dict[tuple[str, str, str, str], ExternalCase] = {(c.dataset, c.case_id, c.ct, c.seg): c for c in cases}
    return sorted(unique.values(), key=lambda c: (c.dataset.lower(), c.case_id.lower())), errors


def _integer_segmentation(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("segmentation must declare millimetre spatial units before physical centroid analysis")
    values = np.asarray(image.get_fdata(dtype=np.float32))
    if values.ndim != 3:
        raise ValueError(f"segmentation must be 3-D: {path} has shape {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError("segmentation must contain finite nonnegative integer labels")
    if values.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("segmentation labels exceed int32 range")
    labels = np.rint(values).astype(np.int32)
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine) != 4:
        raise ValueError("segmentation affine must be finite and invertible")
    return labels, affine, np.asarray(image.header.get_zooms()[:3], dtype=np.float64)


def _label_stats(labels: np.ndarray, affine: np.ndarray) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        coords = np.argwhere(labels == source)
        if len(coords) < 8:
            continue
        centroid = (affine @ np.r_[coords.mean(axis=0), 1.0])[:3]
        result.append({"source_label": source, "voxel_count": int(len(coords)), "centroid_mm": centroid.astype(float).tolist()})
    return result


def map_lumbar_labels(seg_path: str | Path, *, requested_labels: Iterable[int] | None = None) -> dict[str, Any]:
    """Return a deterministic ``L1``--``L5`` source-label mapping.

    The adapter first honours an explicit ``requested_labels`` list, then the
    common 20--24 lumbar ids. Unknown conventions require an explicit verified
    mapping: the five caudal labels can include sacrum or transitional anatomy.
    """
    labels, affine, spacing = _integer_segmentation(seg_path)
    stats = _label_stats(labels, affine)
    present = {int(item["source_label"]) for item in stats}
    explicit = list(requested_labels) if requested_labels is not None else []
    method = ""
    confidence = "high"
    selected: list[int]
    if requested_labels is not None:
        if any(not np.isfinite(v) or int(v) != v for v in explicit):
            raise ValueError("requested_labels must contain integer label IDs")
        explicit = [int(v) for v in explicit]
        if len(explicit) != 5 or len(set(explicit)) != 5 or not set(explicit).issubset(present):
            missing = [v for v in explicit if v not in present]
            raise ValueError(f"requested_labels must contain five distinct present labels; missing={missing}, present={sorted(present)}")
        selected, method = explicit, "explicit"
    elif set(range(20, 25)).issubset(present):
        selected, method = list(range(20, 25)), "standard_lumbar_ids_20_24"
    else:
        raise ValueError(f"cannot identify L1--L5 from labels {sorted(present)}; supply an anatomically verified requested_labels mapping")
    selected_stats = {int(item["source_label"]): item for item in stats}
    mapping = {f"L{i + 1}": int(source) for i, source in enumerate(selected)}
    return {
        "segmentation": str(Path(seg_path).resolve()),
        "voxel_spacing_mm": spacing.astype(float).tolist(),
        "labels_present": sorted(present),
        "label_stats": stats,
        "canonical_to_source": mapping,
        "source_to_canonical": {str(source): f"L{i + 1}" for i, source in enumerate(selected)},
        "selected_source_labels": selected,
        "selection_method": method,
        "selection_confidence": confidence,
        "selected_label_stats": [selected_stats[source] for source in selected],
        "world_coordinate_note": "centroids use the NIfTI affine; L1--L5 identity comes from standard IDs or the explicit supplied mapping",
    }

