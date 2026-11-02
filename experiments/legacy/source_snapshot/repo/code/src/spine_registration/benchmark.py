"""Shared geometry, linear DRR objective and physical evaluation for A/B/C."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from .drr import Volume, params_to_transform
from .projection import project_points
from .relative_metrics import relative_pose_error
from .torch_registration import (TorchConeBeamRenderer, local_motion_tensor,
                                 ncc_loss_torch, motion_penalty_torch, se3_exp)

LABELS = (20, 21, 22, 23, 24)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def load_case(directory, native_paths=None):
    directory = Path(directory)
    truth = json.loads((directory / "motion_ground_truth.json").read_text())
    raw = truth["case"]
    paths = raw if native_paths is None else native_paths
    ct, segmentation = nib.load(paths["ct"]), nib.load(paths["segmentation"])
    if any(image.header.get_xyzt_units()[0] != "mm" for image in (ct, segmentation)):
        raise ValueError("Explicit millimetre NIfTI units are required")
    if ct.shape != segmentation.shape or not np.allclose(ct.affine, segmentation.affine, atol=1e-4, rtol=0):
        raise ValueError(f"{directory.name}: CT and labels do not share a physical grid")
    for affine in (ct.affine, segmentation.affine):
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
            raise ValueError("Nonfinite or singular NIfTI affine")
    with np.load(directory / "centred_case.npz", allow_pickle=False) as archive:
        attenuation = archive["volume"].astype(np.float32)
        seg = archive["segmentation"]
        affine = archive["affine_mm"].astype(float)
        spacing = archive["spacing_mm"].astype(float)
    if (attenuation.shape != seg.shape or not np.isfinite(attenuation).all()
            or not np.isfinite(seg).all() or not np.equal(seg, np.round(seg)).all()
            or (attenuation < 0).any()):
        raise ValueError("Invalid cached attenuation or segmentation")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10 or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Invalid cached physical grid")
    if not np.allclose(affine, raw["centred_affine_mm"], atol=1e-8):
        raise ValueError("Cached affine differs from GT metadata")
    components, centres, params, landmarks, corners = {}, [], [], [], []
    for label in LABELS:
        mask = seg == label
        indices = np.argwhere(mask)
        if len(indices) < 16:
            raise ValueError(f"Missing/empty vertebra {label}")
        centre = affine[:3, :3] @ indices.mean(0) + affine[:3, 3]
        if not np.allclose(centre, truth["centres_mm"][str(label)], atol=1e-5):
            raise ValueError("Segmentation centroid differs from recorded centroid")
        centres.append(centre)
        lo, hi = np.maximum(indices.min(0) - 3, 0), np.minimum(indices.max(0) + 4, seg.shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        cropped_affine = affine.copy()
        cropped_affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
        components[label] = Volume(attenuation[slices] * mask[slices], spacing, affine_mm=cropped_affine)
        edge = np.argwhere(mask & ~binary_erosion(mask))
        chosen = edge[np.linspace(0, len(edge) - 1, min(1000, len(edge)), dtype=int)]
        landmarks.append(chosen @ affine[:3, :3].T + affine[:3, 3])
        bounds = np.array(np.meshgrid(*zip(indices.min(0), indices.max(0)), indexing="ij")).reshape(3, -1).T
        corners.append(bounds @ affine[:3, :3].T + affine[:3, 3])
        motion = truth["motions"][str(label)]
        params.append([*np.deg2rad([motion[k] for k in ["rx_deg", "ry_deg", "rz_deg"]]),
                       *[motion[k] for k in ["tx_mm", "ty_mm", "tz_mm"]]])
    for i, label in enumerate(LABELS):
        expected = params_to_transform(params[i])
        expected[:3, 3] += centres[i] - expected[:3, :3] @ centres[i]
        saved = np.asarray(truth["transforms_mm"][str(label)], dtype=float)
        if saved.shape != (4, 4) or not np.isfinite(saved).all() or not np.allclose(saved, expected, atol=1e-8):
            raise ValueError("GT parameter/matrix inconsistency")
    return {"case": directory.name, "components": components, "centres": np.array(centres),
            "gt_params": np.array(params), "landmarks": landmarks, "corners": corners,
            "gt_matrices": np.array([truth["transforms_mm"][str(k)] for k in LABELS]),
            "native_grid_checked": True}


def projections(pixel_spacing_mm=2.0):
    # Both detector vertical axes point inferiorly; world z is superior.
    rotations = {"AP": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
                 "LAT": np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)}
    intrinsic = np.array([[1200 / pixel_spacing_mm, 0, 127.5],
                          [0, 1200 / pixel_spacing_mm, 127.5], [0, 0, 1]])
    return {v: intrinsic @ np.column_stack([r, [0, 0, 750]]) for v, r in rotations.items()}


def global_to_local(pose, centres):
    matrix = params_to_transform(pose)
    out = np.tile(pose, (len(centres), 1)).copy()
    out[:, 3:] = centres @ matrix[:3, :3].T + matrix[:3, 3] - centres
    return out


def evaluate(params, case, method):
    params = np.asarray(params, dtype=float)
    if method == "A":
        params = np.tile(params.reshape(6), (5, 1))
    elif method == "Zero":
        params = np.zeros((5, 6))
    else:
        params = params.reshape(5, 6)
    gt, centres = case["gt_params"], case["centres"]
    vertebrae, relative = [], []
    for i, centre in enumerate(centres):
        estimate = params_to_transform(params[i])
        if method != "A":
            estimate[:3, 3] += centre - estimate[:3, :3] @ centre
        truth = case["gt_matrices"][i]
        point = np.r_[centre, 1.0]
        delta = (estimate @ point - truth @ point)[:3]
        angle = np.degrees(Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude())
        surface = case["landmarks"][i]
        moved_est = surface @ estimate[:3, :3].T + estimate[:3, 3]
        moved_gt = surface @ truth[:3, :3].T + truth[:3, 3]
        vertebrae.append({"label": LABELS[i], "translation_error_mm": float(np.linalg.norm(delta)),
                          "rotation_error_deg": float(angle), "axis_error_mm": np.abs(delta).tolist(),
                          "surface_mtre_mm": float(np.linalg.norm(moved_est - moved_gt, axis=1).mean())})
        if i < 4:
            relative.append(relative_pose_error(gt[i], gt[i + 1], params[i], params[i + 1],
                centre, centres[i + 1], method="rigid_global" if method == "A" else "independent_vertebrae"))
    metrics = {key: float(np.mean([v[key] for v in vertebrae])) for key in
               ["translation_error_mm", "rotation_error_deg", "surface_mtre_mm"]}
    metrics.update({key: float(np.mean([v[key] for v in relative])) for key in
                    ["relative_translation_error_mm", "relative_rotation_error_deg"]})
    errors = np.array([v["translation_error_mm"] for v in vertebrae])
    metrics.update({"vertebra_failure_rate_gt3mm": float(np.mean(errors > 3)),
                    "case_failure_gt3mm": bool((errors > 3).any()),
                    "case_mean_gt3mm": bool(errors.mean() > 3)})
    return {"metrics": metrics, "vertebrae": vertebrae, "relative_edges": relative}


class ImageProblem:
    def __init__(self, case, config, device="cuda"):
        self.case, self.config, self.device = case, config, device
        self.centres = torch.as_tensor(case["centres"], device=device, dtype=torch.float32)
        self.projections = projections(config["pixel_spacing_mm"])
        self.renderers, self.targets = {}, {}
        for points in case["corners"]:
            for projection in self.projections.values():
                pixels = project_points(points, projection)
                if pixels.min() < 6 or pixels.max() > 249:
                    raise ValueError(f"{case['case']}: anatomy outside detector safety margin")

    def ensure_level(self, level):
        if level in self.renderers:
            return
        s = level / 256
        scale = np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1]])
        self.renderers[level] = {view: [TorchConeBeamRenderer(self.case["components"][label], scale @ projection,
            (level, level), device=self.device, sample_step_mm=self.config["sample_step_mm"],
            max_samples=self.config["max_samples"], chunk_rows=min(level, self.config["chunk_rows"]))
            for label in LABELS] for view, projection in self.projections.items()}

    def parts(self, params, level, global_pose=False):
        self.ensure_level(level)
        transforms = se3_exp(params).expand(5, 4, 4) if global_pose else local_motion_tensor(params, self.centres)
        return {view: [r.render(transforms[i], log_transform=False, normalize_output=False)
                       for i, r in enumerate(renderers)] for view, renderers in self.renderers[level].items()}

    def synthesize(self, params):
        with torch.no_grad():
            parts = self.parts(torch.as_tensor(params, device=self.device, dtype=torch.float32), 256)
            return {view: torch.stack(images).sum(0) for view, images in parts.items()}, parts

    def set_targets(self, targets):
        for level in self.config["levels"]:
            self.targets[level] = {v: F.interpolate(t[None, None], size=(level, level), mode="area")[0, 0]
                                   for v, t in targets.items()}

    def image_loss(self, params, level, global_pose=False):
        images = self.parts(params, level, global_pose)
        return torch.stack([ncc_loss_torch(self.targets[level][v], torch.stack(parts).sum(0))
                            for v, parts in images.items()]).mean()

    def joint_loss(self, params, level):
        return self.image_loss(params, level) + self.config["lambda_motion"] * motion_penalty_torch(
            params, centres=self.centres, sigma_rotation_deg=3.0, sigma_translation_mm=3.0)

    def block_objective(self, index, frozen, level):
        with torch.no_grad():
            parts = self.parts(frozen, level)
            others = {v: torch.stack([image for j, image in enumerate(images) if j != index]).sum(0)
                      for v, images in parts.items()}
        def objective(pose):
            matrix = local_motion_tensor(pose, self.centres[index])
            return torch.stack([ncc_loss_torch(self.targets[level][view], others[view] +
                self.renderers[level][view][index].render(matrix, log_transform=False, normalize_output=False))
                for view in self.projections]).mean()
        return objective

    def powell_objective(self, level):
        self.ensure_level(level)
        def objective(params):
            with torch.no_grad():
                return float(self.image_loss(torch.as_tensor(params, device=self.device, dtype=torch.float32), level, True))
        return objective


def occlude(targets, gt_parts, endpoint, seed, fraction=0.30):
    """GT projection defines synthetic corruption only, never registration input masks."""
    rng = np.random.default_rng(seed)
    result, metadata = {}, {}
    for view, target in targets.items():
        part = gt_parts[view][endpoint].cpu().numpy()
        mask = part > .01 * part.max()
        ys, xs = np.nonzero(mask)
        if len(xs) < 10:
            raise ValueError("Endpoint not sufficiently visible for occlusion")
        best = None
        for _ in range(1000):
            width = int(rng.integers(max(1, int(np.ptp(xs) * .3)), max(2, int(np.ptp(xs) * .9))))
            height = max(1, min(256, round(fraction * len(xs) / width)))
            x = int(np.clip(rng.integers(xs.min(), xs.max() + 1) - width // 2, 0, 256 - width))
            y = int(np.clip(rng.integers(ys.min(), ys.max() + 1) - height // 2, 0, 256 - height))
            covered = float(mask[y:y + height, x:x + width].sum() / len(xs))
            if best is None or abs(covered - fraction) < best[0]:
                best = (abs(covered - fraction), x, y, width, height, covered)
        _, x, y, width, height, covered = best
        image = target.clone()
        image[y:y + height, x:x + width] = 0
        result[view] = image
        metadata[view] = {"rectangle_xywh": [x, y, width, height], "endpoint_coverage": covered,
                          "endpoint_label": LABELS[endpoint], "seed": seed}
    return result, metadata
