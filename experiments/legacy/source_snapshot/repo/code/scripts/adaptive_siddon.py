"""Independent, exact voxel-cell line integrals for synthetic sensitivity tests.

Each array value is constant inside a voxel centred on its integer index, with
faces at half-integers. This deliberately differs from the benchmark renderer's
trilinear interpolation and centre-to-centre support. Pixels use the existing
positive-camera-depth pinhole convention. The model still shares synthetic
attenuation physics; it is neither a clinical simulator nor a differentiable
registration renderer.
"""
from __future__ import annotations

import numpy as np
import torch

from spine_registration.drr import Volume


@torch.no_grad()
def render(volume: Volume, projection, transform=None, image_shape=(256, 256),
           device="cuda", chunk_rays=1024) -> torch.Tensor:
    """Return raw attenuation integrals in physical mm, as a float64 image.

    ``transform`` maps the volume's physical coordinates to the camera world frame.
    Geometry and integration use float64 on both CPU and GPU. No interpolation,
    sampling step, log transform, output normalization, or benchmark renderer is used.
    Memory per chunk scales with ``chunk_rays * sum(volume.data.shape)``.
    """
    data = np.asarray(volume.data)
    if data.ndim != 3 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError("volume must be a nonempty finite 3-D array")
    if (len(image_shape) != 2 or any(int(n) != n or n < 1 for n in image_shape)
            or int(chunk_rays) != chunk_rays or chunk_rays < 1):
        raise ValueError("image_shape and chunk_rays must contain positive integers")
    height, width = map(int, image_shape)
    chunk_rays = int(chunk_rays)
    p = np.asarray(projection, dtype=np.float64)
    if p.shape != (3, 4) or not np.isfinite(p).all():
        raise ValueError("projection must be a finite 3x4 matrix")
    if isinstance(transform, torch.Tensor):
        transform = transform.detach().cpu().numpy()
    pose = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError("transform must be a finite rigid homogeneous 4x4 matrix")
    affine = volume.voxel_to_world_affine()
    world_to_voxel = np.linalg.inv(pose @ affine)
    v, u = np.mgrid[:height, :width]
    pixels = np.column_stack((u.ravel(), v.ravel(), np.ones(height * width)))
    try:
        # P = M [I | -C]: its null point is C, and M^-1 [u,v,1] is a ray.
        source_world = np.linalg.solve(p[:, :3], -p[:, 3])
        directions_world = np.linalg.solve(p[:, :3], pixels.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("projection must have a nonsingular left 3x3 block") from exc
    norms = np.linalg.norm(directions_world, axis=1, keepdims=True)
    if (not np.isfinite(source_world).all() or not np.isfinite(norms).all()
            or np.any(norms == 0)):
        raise ValueError("projection produces nonfinite or zero-length rays")
    directions_world /= norms
    source_local = world_to_voxel[:3, :3] @ source_world + world_to_voxel[:3, 3]
    directions_local = directions_world @ world_to_voxel[:3, :3].T
    if not np.isfinite(source_local).all() or not np.isfinite(directions_local).all():
        raise ValueError("volume affine and transform produce nonfinite rays")
    source = torch.as_tensor(source_local, dtype=torch.float64, device=device)
    directions = torch.as_tensor(directions_local, dtype=torch.float64, device=device)
    voxels = torch.as_tensor(data, dtype=torch.float64, device=device)
    upper = source.new_tensor(data.shape) - 0.5
    planes = [torch.arange(n + 1, dtype=source.dtype, device=device) - 0.5
              for n in data.shape]
    output = source.new_zeros(height * width)
    for start in range(0, height * width, chunk_rays):
        ray = directions[start:start + chunk_rays]
        parallel = ray == 0
        divisor = torch.where(parallel, 1.0, ray)
        first, last = (-0.5 - source) / divisor, (upper - source) / divisor
        entry = torch.where(parallel, -torch.inf, torch.minimum(first, last))
        exit_ = torch.where(parallel, torch.inf, torch.maximum(first, last))
        near, far = entry.amax(dim=1).clamp_min(0), exit_.amin(dim=1)
        outside = (parallel & ((source < -0.5) | (source > upper))).any(dim=1)
        valid = ~outside & torch.isfinite(near) & torch.isfinite(far) & (far > near)
        ray, divisor, parallel = ray[valid], divisor[valid], parallel[valid]
        near, far = near[valid, None], far[valid, None]
        crossings = [near, far]
        for axis, faces in enumerate(planes):
            times = (faces[None] - source[axis]) / divisor[:, axis, None]
            times = torch.where(parallel[:, axis, None], near, times)
            crossings.append(times.maximum(near).minimum(far))
        times = torch.cat(crossings, dim=1).sort(dim=1).values
        lengths = times[:, 1:] - times[:, :-1]
        midpoints = (times[:, 1:] + times[:, :-1]) * 0.5
        indices = torch.floor(source + midpoints[..., None] * ray[:, None] + 0.5).long()
        # Clipping only matters for duplicate, zero-length intervals at box faces.
        values = voxels[tuple(indices[..., axis].clamp(0, n - 1)
                              for axis, n in enumerate(data.shape))]
        output[start:start + chunk_rays][valid] = (values * lengths).sum(dim=1)
    return output.reshape(height, width)
