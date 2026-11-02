"""Isolated geometric posture simulation; angles are supplied by the experiment.

Midpoint hinges are anatomical proxies, not identified disc centres. Distances
use dense boundary voxel centres and therefore retain voxel-scale uncertainty.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


def make_articulated_motion(centres, segment_angles_deg, global_rotvec_rad,
                            global_translation_mm):
    """L5-anchored RAS-x chain, followed by a global motion about L5.

    Angles and pivots are ordered L1/L2 through L4/L5. ``params`` contains
    rotation vectors and centroid displacements, not matrix translation columns.
    """
    centres = np.asarray(centres, dtype=float)
    angles = np.asarray(segment_angles_deg, dtype=float)
    rotation = np.asarray(global_rotvec_rad, dtype=float)
    translation = np.asarray(global_translation_mm, dtype=float)
    if centres.shape != (5, 3) or angles.shape != (4,) or rotation.shape != (3,) or translation.shape != (3,):
        raise ValueError("Expected centres (5,3), segment angles (4,), and global vectors (3,)")
    if not all(np.isfinite(value).all() for value in (centres, angles, rotation, translation)):
        raise ValueError("Motion inputs must be finite")
    pivots = (centres[:-1] + centres[1:]) / 2
    articulated = np.tile(np.eye(4), (5, 1, 1))
    for i in range(3, -1, -1):
        hinge = np.eye(4)
        hinge[:3, :3] = Rotation.from_rotvec([np.deg2rad(angles[i]), 0., 0.]).as_matrix()
        hinge[:3, 3] = pivots[i] - hinge[:3, :3] @ pivots[i]
        articulated[i] = articulated[i + 1] @ hinge
    global_transform = np.eye(4)
    global_transform[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    global_transform[:3, 3] = centres[-1] + translation - global_transform[:3, :3] @ centres[-1]
    transforms = global_transform[None] @ articulated
    moved_centres = (transforms[:, :3, :3] @ centres[..., None]).squeeze(-1) + transforms[:, :3, 3]
    params = np.column_stack([Rotation.from_matrix(transforms[:, :3, :3]).as_rotvec(), moved_centres - centres])
    return dict(transforms=transforms, params=params, pivots=pivots,
                articulated_transforms=articulated, global_transform=global_transform)


def prepare_geometry(segmentation, affine_mm, labels=(20, 21, 22, 23, 24)):
    """Cache cropped masks, physical SDFs and every boundary voxel centre.

    Orthogonal voxel axes may be rotated or reflected; sheared grids require
    resampling before an EDT can represent Euclidean millimetres correctly.
    """
    seg = np.asarray(segmentation)
    affine = np.asarray(affine_mm, dtype=float)
    labels = tuple(int(label) for label in labels)
    if seg.ndim != 3 or not np.isfinite(seg).all() or not np.equal(seg, np.round(seg)).all():
        raise ValueError("Segmentation must be a finite integer-valued 3-D array")
    if len(labels) < 2 or len(set(labels)) != len(labels) or 0 in labels:
        raise ValueError("At least two distinct nonbackground labels are required")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("A finite homogeneous affine is required")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0) or not np.allclose(affine[:3, :3].T @ affine[:3, :3], np.diag(spacing ** 2), atol=1e-8):
        raise ValueError("Distance audit requires nonsingular orthogonal voxel axes")
    components, centres = [], []
    for label in labels:
        indices = np.argwhere(seg == label)
        if not len(indices):
            raise ValueError(f"Missing segmentation label {label}")
        centre = affine[:3, :3] @ indices.mean(0) + affine[:3, 3]
        centres.append(centre)
        lo, hi = indices.min(0), indices.max(0) + 1
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        mask = np.pad(seg[slices] == label, 2)
        cropped_affine = affine.copy()
        cropped_affine[:3, 3] = affine[:3, :3] @ (lo - 2) + affine[:3, 3]
        signed = ndimage.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
        signed -= ndimage.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
        boundary = np.argwhere(mask & ~ndimage.binary_erosion(mask, border_value=0))
        surface = boundary @ cropped_affine[:3, :3].T + cropped_affine[:3, 3]
        # Bounds are voxel-cell faces; overlap sampling itself uses voxel centres.
        corners = np.array(np.meshgrid(*zip(lo - .5, hi - .5), indexing="ij")).reshape(3, -1).T
        components.append(dict(label=label, mask=mask, sdf_mm=signed, world_to_voxel=np.linalg.inv(cropped_affine),
                               surface_mm=surface, corners_mm=corners @ affine[:3, :3].T + affine[:3, 3],
                               voxel_count=int(len(indices))))
    centres = np.asarray(centres)
    if np.any(np.linalg.norm(np.diff(centres, axis=0), axis=1) < 1e-9):
        raise ValueError("Adjacent labels must have distinct centres")
    return dict(labels=labels, centres=centres, affine_mm=affine, spacing_mm=spacing,
                voxel_tolerance_mm=float(spacing.max()), components=components)


def _map_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _penetration(source, neighbour, relative_transform, tolerance):
    indices = _map_points(_map_points(source["surface_mm"], relative_transform), neighbour["world_to_voxel"])
    distances = ndimage.map_coordinates(neighbour["sdf_mm"], indices.T, order=1,
                                        mode="constant", cval=np.inf, prefilter=False)
    depths = np.maximum(-distances, 0)
    maximum = float(depths.max(initial=0))
    return dict(surface_points=int(len(distances)), inside_fraction=float(np.mean(distances < 0)),
                penetration_fraction_beyond_tolerance=float(np.mean(depths > tolerance)),
                max_depth_mm=maximum, max_excess_depth_mm=max(0., maximum - tolerance))


def _overlap_on_common_grid(prepared, first, second, transform_i, transform_j):
    affine = prepared["affine_mm"]
    world_to_grid = np.linalg.inv(affine)
    bounds_i = _map_points(first["corners_mm"], world_to_grid @ transform_i)
    bounds_j = _map_points(second["corners_mm"], world_to_grid @ transform_j)
    lower = np.ceil(np.maximum(bounds_i.min(0), bounds_j.min(0))).astype(np.int64)
    upper = np.floor(np.minimum(bounds_i.max(0), bounds_j.max(0))).astype(np.int64)
    shape = upper - lower + 1
    count = 0
    if np.all(shape > 0):
        total = int(np.prod(shape))
        inverse_i = first["world_to_voxel"] @ np.linalg.inv(transform_i) @ affine
        inverse_j = second["world_to_voxel"] @ np.linalg.inv(transform_j) @ affine
        # A bounded slab avoids allocating a float64 coordinate grid for the volume.
        for start in range(0, total, 131072):
            flat = np.arange(start, min(total, start + 131072))
            points = np.column_stack(np.unravel_index(flat, tuple(shape))) + lower
            a = ndimage.map_coordinates(first["mask"], _map_points(points, inverse_i).T,
                                        order=0, mode="constant", cval=0, prefilter=False)
            b = ndimage.map_coordinates(second["mask"], _map_points(points, inverse_j).T,
                                        order=0, mode="constant", cval=0, prefilter=False)
            count += int(np.count_nonzero(a & b))
    return dict(voxels=count, volume_mm3=float(count * abs(np.linalg.det(affine[:3, :3]))),
                fraction_smaller_volume=float(count / min(first["voxel_count"], second["voxel_count"])))


def audit_geometry(prepared, transforms, pivots=None, include_overlap=False):
    """Report geometric measurements; acceptance thresholds belong to the study.

    Tolerance is one largest-axis voxel spacing. The audit checks adjacent
    labelled bones, not discs, ligaments, canal geometry or mechanical forces.
    """
    transforms = np.asarray(transforms, dtype=float)
    centres = prepared["centres"]
    count = len(centres)
    if transforms.shape != (count, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Expected one finite 4x4 rigid matrix per prepared component")
    rotations = transforms[:, :3, :3]
    if (not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)):
        raise ValueError("Audit transforms must be proper rigid motions")
    pivots = (centres[:-1] + centres[1:]) / 2 if pivots is None else np.asarray(pivots, dtype=float)
    if pivots.shape != (count - 1, 3) or not np.isfinite(pivots).all():
        raise ValueError("Expected one finite pivot per adjacent pair")
    inverse = np.linalg.inv(transforms)
    moved = (rotations @ centres[..., None]).squeeze(-1) + transforms[:, :3, 3]
    tolerance = prepared["voxel_tolerance_mm"]
    edges = []
    for i in range(count - 1):
        j = i + 1
        first, second = prepared["components"][i], prepared["components"][j]
        left = _penetration(first, second, inverse[j] @ transforms[i], tolerance)
        right = _penetration(second, first, inverse[i] @ transforms[j], tolerance)
        distance = float(np.linalg.norm(moved[i] - moved[j]))
        edges.append(dict(labels=[int(first["label"]), int(second["label"])], i_in_j=left, j_in_i=right,
                          centroid_distance_mm=distance,
                          centroid_distance_ratio=distance / float(np.linalg.norm(centres[i] - centres[j])),
                          hinge_residual_mm=float(np.linalg.norm(_map_points(pivots[i], transforms[i]) - _map_points(pivots[i], transforms[j]))),
                          overlap=_overlap_on_common_grid(prepared, first, second, transforms[i], transforms[j]) if include_overlap else None))
    directions = [edge[key] for edge in edges for key in ("i_in_j", "j_in_i")]
    return dict(voxel_tolerance_mm=tolerance, edges=edges,
                max_penetration_depth_mm=max(value["max_depth_mm"] for value in directions),
                max_excess_depth_mm=max(value["max_excess_depth_mm"] for value in directions),
                max_penetration_fraction_beyond_tolerance=max(value["penetration_fraction_beyond_tolerance"] for value in directions),
                max_hinge_residual_mm=max(edge["hinge_residual_mm"] for edge in edges),
                max_overlap_fraction_smaller_volume=max(edge["overlap"]["fraction_smaller_volume"] for edge in edges) if include_overlap else None)
