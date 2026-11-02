"""Differentiable CUDA cone-beam DRR and articulated Method-C registration.

The NumPy/SciPy implementation remains the reference baseline.  This module
keeps the same physical-mm convention and the same relative SE(3) spring used
by ``motion_methods.motion_penalty`` while replacing finite-search rendering
with ``grid_sample`` and gradient-based optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F

from .drr import Volume, _integration_sample_count, _rays_from_projection_matrix


def _as_tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _hat(v: Tensor) -> Tensor:
    """Batch skew matrices for vectors with final dimension three."""
    out = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def so3_exp(rotvec: Tensor) -> Tensor:
    """Differentiable Rodrigues exponential for ``(..., 3)`` vectors."""
    omega = rotvec
    theta2 = torch.sum(omega * omega, dim=-1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta2, min=torch.finfo(omega.dtype).tiny))
    k = _hat(omega)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    # sinc avoids cancellation in 1-cos(theta) and has an analytic zero limit.
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * torch.pi)).square()
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def pose_matrix(params: Tensor) -> Tensor:
    """Map rotation vectors (rad) and direct translations (mm) to SE(3).

    The last three parameters are physical translations, not the velocity
    coordinates of an se(3) twist; no SO(3) left Jacobian is applied to them.
    """
    params = params.reshape(*params.shape[:-1], 6)
    out = torch.eye(4, device=params.device, dtype=params.dtype).expand(*params.shape[:-1], 4, 4).clone()
    out[..., :3, :3] = so3_exp(params[..., :3])
    out[..., :3, 3] = params[..., 3:]
    return out


def se3_exp(params: Tensor) -> Tensor:
    """Compatibility alias for :func:`pose_matrix`, not a twist exponential."""
    return pose_matrix(params)


def se3_log(transform: Tensor) -> Tensor:
    """Principal SE(3) logarithm with stable identity and near-pi limits."""
    r = transform[..., :3, :3]
    t = transform[..., :3, 3]
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    qabs = torch.sqrt(torch.stack((1 + r00 + r11 + r22, 1 + r00 - r11 - r22,
                                  1 - r00 + r11 - r22, 1 - r00 - r11 + r22), dim=-1).clamp_min(1e-12))
    x, y, z = r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]
    xy, xz, yz = r[..., 1, 0] + r[..., 0, 1], r[..., 2, 0] + r[..., 0, 2], r[..., 2, 1] + r[..., 1, 2]
    candidates = torch.stack((torch.stack((qabs[..., 0].square(), x, y, z), -1),
                              torch.stack((x, qabs[..., 1].square(), xy, xz), -1),
                              torch.stack((y, xy, qabs[..., 2].square(), yz), -1),
                              torch.stack((z, xz, yz, qabs[..., 3].square()), -1)), -2)
    candidates = candidates / (2 * qabs.clamp_min(0.1).unsqueeze(-1))
    index = qabs.argmax(-1)[..., None, None].expand(*qabs.shape[:-1], 1, 4)
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    sine_half2 = quaternion[..., 1:].square().sum(-1, keepdim=True)
    sine_half = sine_half2.clamp_min(1e-16).sqrt()
    theta = 2 * torch.atan2(sine_half, quaternion[..., :1])
    factor = torch.where(sine_half2 < 1e-6, 2 + sine_half2 / 3 + 3 * sine_half2.square() / 20,
                         theta / sine_half)
    omega = factor * quaternion[..., 1:]
    omega_hat = _hat(omega)
    theta2 = omega.square().sum(-1, keepdim=True)
    safe_theta = theta2.clamp_min(1e-4).sqrt()
    coefficient = torch.where(theta2 < 1e-2, 1 / 12 + theta2 / 720 + theta2.square() / 30240,
                              (1 - 0.5 * safe_theta / torch.tan(0.5 * safe_theta)) / safe_theta.square())
    v_inv = torch.eye(3, device=transform.device, dtype=transform.dtype).expand_as(omega_hat) - 0.5 * omega_hat + coefficient[..., None] * (omega_hat @ omega_hat)
    return torch.cat((omega, (v_inv @ t.unsqueeze(-1)).squeeze(-1)), dim=-1)


def local_motion_tensor(params: Tensor, centre_mm: Tensor) -> Tensor:
    """The same centre-localized motion as ``motion_methods.local_motion``."""
    centre = centre_mm.to(params)
    transform = pose_matrix(params)
    transform[..., :3, 3] = centre + params[..., 3:] - (transform[..., :3, :3] @ centre.unsqueeze(-1)).squeeze(-1)
    return transform


def ncc_loss_torch(reference: Tensor, moving: Tensor, eps: float = 1e-12) -> Tensor:
    """Differentiable one-minus zero-mean normalized cross correlation."""
    if reference.shape != moving.shape or reference.numel() == 0:
        raise ValueError("NCC images must have the same nonempty shape")
    a = reference.reshape(-1)
    b = moving.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    valid = norm_a * norm_b >= eps
    safe_a = torch.where(valid, norm_a, torch.ones_like(norm_a))
    safe_b = torch.where(valid, norm_b, torch.ones_like(norm_b))
    # Equivalent to 1-NCC, without float32 cancellation near perfect alignment.
    loss = 0.5 * (a / safe_a - b / safe_b).square().sum()
    return torch.where(valid, loss, torch.ones_like(loss))


@dataclass
class _RayBundle:
    source: Tensor
    directions: Tensor
    pixels: tuple[int, int]


class TorchConeBeamRenderer:
    """GPU cone-beam renderer matching ``render_cone_beam_drr`` geometry."""

    def __init__(self, volume: Volume | np.ndarray, projection_matrix: np.ndarray,
                 output_size: tuple[int, int], *, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32, chunk_rows: int = 32,
                 sample_step_mm: float | None = None, max_samples: int = 256):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.dtype = dtype
        self.volume = volume if isinstance(volume, Volume) else Volume(np.asarray(volume, dtype=np.float32), np.ones(3, dtype=np.float32))
        if self.volume.data.ndim != 3 or min(self.volume.data.shape) < 2:
            raise ValueError("volume must be 3-D")
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.chunk_rows = int(chunk_rows)
        self.max_samples = int(max_samples)
        self.sample_step_mm = sample_step_mm
        if min(self.output_size) < 2 or self.chunk_rows < 1 or self.max_samples < 2:
            raise ValueError("invalid output_size, chunk_rows or max_samples")
        if not np.isfinite(self.volume.data).all():
            raise ValueError("volume contains NaN or Inf")
        self.n_samples = _integration_sample_count(self.volume, self.volume.voxel_to_world_affine(), sample_step_mm, self.max_samples)
        self.fractions = (torch.arange(self.n_samples, device=self.device, dtype=self.dtype) + 0.5) / self.n_samples
        self.data = torch.as_tensor(self.volume.data, device=self.device, dtype=self.dtype)[None, None]
        self.voxel_to_world = _as_tensor(self.volume.voxel_to_world_affine(), device=self.device, dtype=self.dtype)
        self.world_to_voxel = torch.linalg.inv(self.voxel_to_world)
        self.projection_matrix = np.asarray(projection_matrix, dtype=np.float64).reshape(3, 4)
        if not np.isfinite(self.projection_matrix).all() or np.linalg.matrix_rank(self.projection_matrix) < 3:
            raise ValueError("projection_matrix must be finite and rank 3")
        self.rays = self._make_rays()

    def _make_rays(self) -> _RayBundle:
        h, w = self.output_size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64), indexing="xy")
        pixels = np.c_[u.ravel(), v.ravel()]
        source, directions = _rays_from_projection_matrix(self.projection_matrix, pixels)
        return _RayBundle(_as_tensor(source, device=self.device, dtype=self.dtype),
                          _as_tensor(directions, device=self.device, dtype=self.dtype), (h, w))

    def render(self, transform_mm: Tensor | np.ndarray | None = None, *, log_transform: bool = False,
               normalize_output: bool = False) -> Tensor:
        """Render one image; gradients flow to ``transform_mm``."""
        if transform_mm is None:
            transform = torch.eye(4, device=self.device, dtype=self.dtype)
        elif isinstance(transform_mm, Tensor):
            transform = transform_mm.to(device=self.device, dtype=self.dtype)
        else:
            transform = _as_tensor(transform_mm, device=self.device, dtype=self.dtype)
        if transform.shape != (4, 4):
            raise ValueError("transform_mm must have shape (4, 4)")
        world_to_local = self.world_to_voxel @ torch.linalg.inv(transform)
        source = world_to_local[:3, :3] @ self.rays.source + world_to_local[:3, 3]
        upper = torch.as_tensor(self.volume.data.shape, device=self.device, dtype=self.dtype) - 1
        outputs = []
        h, w = self.output_size
        n_samples = self.n_samples
        for row_start in range(0, h, self.chunk_rows):
            row_stop = min(h, row_start + self.chunk_rows)
            directions = self.rays.directions[row_start * w:row_stop * w] @ world_to_local[:3, :3].T
            parallel = directions.abs() < 1e-12
            divisor = torch.where(parallel, torch.ones_like(directions), directions)
            t0, t1 = -source[None] / divisor, (upper - source)[None] / divisor
            outside = (parallel & ((source < 0) | (source > upper))[None]).any(dim=-1)
            t0 = torch.where(parallel, -torch.inf, t0)
            t1 = torch.where(parallel, torch.inf, t1)
            near = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0)
            far = torch.maximum(t0, t1).amin(dim=-1)
            valid = ~outside & torch.isfinite(near) & torch.isfinite(far) & (far > near)
            near = torch.where(valid, near, torch.zeros_like(near))
            lengths = torch.where(valid, far - near, torch.zeros_like(far))
            distances = near[:, None] + lengths[:, None] * self.fractions[None, :]
            indices = (source[None, None, :] + distances[..., None] * directions[:, None, :]).reshape(
                1, row_stop - row_start, w * n_samples, 1, 3)
            # grid_sample expects x/y/z = last/second/first voxel axes.
            d, hh, ww = self.volume.data.shape
            grid = torch.stack((indices[..., 2] / max(ww - 1, 1) * 2.0 - 1.0,
                                indices[..., 1] / max(hh - 1, 1) * 2.0 - 1.0,
                                indices[..., 0] / max(d - 1, 1) * 2.0 - 1.0), dim=-1)
            # Reshape as [N, D_out, H_out, W_out, 3] for 5-D sampling, then
            # flatten the sample dimension back onto each ray.
            grid = grid.reshape(1, row_stop - row_start, w * n_samples, 1, 3)
            samples = F.grid_sample(self.data, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            samples = samples[0, 0, :, :, 0].reshape((row_stop - row_start) * w, n_samples)
            samples = torch.where(valid[:, None], samples, torch.zeros_like(samples))
            line = samples.mean(dim=-1) * lengths
            if log_transform:
                line = torch.log1p(line.clamp_min(0.0))
            outputs.append(line.reshape(row_stop - row_start, w))
        image = torch.cat(outputs, dim=0)
        if normalize_output:
            lo, hi = torch.amin(image), torch.amax(image)
            image = (image - lo) / (hi - lo).clamp_min(torch.finfo(image.dtype).eps)
        return image


def motion_penalty_torch(params: Tensor, labels: Iterable[int] | None = None,
                         sigma_rotation_deg: float = 3.0, sigma_translation_mm: float = 3.0,
                         centres: Tensor | None = None) -> Tensor:
    """PyTorch equivalent of ``motion_methods.motion_penalty``."""
    del labels
    params = params.reshape(-1, 6)
    if sigma_rotation_deg <= 0 or sigma_translation_mm <= 0:
        raise ValueError("motion-prior scales must be positive")
    if params.shape[0] < 2:
        return params.sum() * 0.0
    if centres is None:
        transforms = se3_exp(params)
        deltas = torch.linalg.inv(transforms[:-1]) @ transforms[1:]
    else:
        centres = centres.to(device=params.device, dtype=params.dtype).reshape(-1, 3)
        centre_transforms = torch.eye(4, device=params.device, dtype=params.dtype).expand(params.shape[0], 4, 4).clone()
        centre_transforms[:, :3, 3] = centres
        inverse_centres = torch.eye(4, device=params.device, dtype=params.dtype).expand(params.shape[0], 4, 4).clone()
        inverse_centres[:, :3, 3] = -centres
        local = centre_transforms @ se3_exp(params) @ inverse_centres
        post = local @ centre_transforms
        pre = centre_transforms
        d_pre = torch.linalg.inv(pre[:-1]) @ pre[1:]
        d_post = torch.linalg.inv(post[:-1]) @ post[1:]
        deltas = torch.linalg.inv(d_pre) @ d_post
    twist = se3_log(deltas)
    rot = torch.sum((twist[:, :3] / max(np.deg2rad(float(sigma_rotation_deg)), 1e-6)) ** 2, dim=-1)
    trans = torch.sum((twist[:, 3:] / max(float(sigma_translation_mm), 1e-6)) ** 2, dim=-1)
    return (rot + trans).mean()

class TorchMethodCRegistration:
    """Joint five-vertebra Method C optimizer with one 30-D Parameter."""

    def __init__(self, components: Mapping[int, Volume | np.ndarray], centres_mm: Mapping[int, Iterable[float]],
                 projections: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray],
                 output_size: tuple[int, int], *, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32, sample_step_mm: float = 1.5,
                 max_samples: int = 96, chunk_rows: int = 32, lambda_motion: float = 0.1,
                 sigma_rotation_deg: float = 3.0, sigma_translation_mm: float = 3.0):
        self.device, self.dtype = torch.device(device), dtype
        self.labels = sorted(components)
        if len(self.labels) != 5:
            raise ValueError(f"Method C expects five vertebrae, got {len(self.labels)}")
        if set(projections) != set(targets) or not projections:
            raise ValueError("projections and targets must contain the same nonempty views")
        if any(np.shape(image) != output_size or not np.isfinite(image).all() for image in targets.values()):
            raise ValueError("targets must be finite images matching output_size")
        if not np.isfinite(lambda_motion) or lambda_motion < 0:
            raise ValueError("lambda_motion must be finite and nonnegative")
        self.centres = torch.stack([_as_tensor(centres_mm[k], device=self.device, dtype=dtype) for k in self.labels])
        self.renderers = {view: [TorchConeBeamRenderer(components[k], p, output_size, device=self.device, dtype=dtype, chunk_rows=chunk_rows, sample_step_mm=sample_step_mm, max_samples=max_samples) for k in self.labels] for view, p in projections.items()}
        self.targets = {view: _as_tensor(np.asarray(image), device=self.device, dtype=dtype) for view, image in targets.items()}
        self.lambda_motion = float(lambda_motion)
        self.sigma_rotation_deg = float(sigma_rotation_deg)
        self.sigma_translation_mm = float(sigma_translation_mm)
        self.params = Parameter(torch.zeros((5, 6), device=self.device, dtype=dtype))

    def loss(self) -> tuple[Tensor, dict[str, Tensor]]:
        view_losses = []
        for view, renderers in self.renderers.items():
            prediction = torch.stack([renderer.render(local_motion_tensor(self.params[i], self.centres[i]), log_transform=False, normalize_output=False) for i, renderer in enumerate(renderers)]).sum(dim=0)
            view_losses.append(ncc_loss_torch(self.targets[view], prediction))
        image_loss = torch.stack(view_losses).mean()
        penalty = motion_penalty_torch(self.params, sigma_rotation_deg=self.sigma_rotation_deg,
                                       sigma_translation_mm=self.sigma_translation_mm,
                                       centres=self.centres)
        return image_loss + self.lambda_motion * penalty, {"image": image_loss, "motion": penalty}

    def optimize(self, *, initial_params: np.ndarray | None = None, adam_steps: int = 100,
                 adam_lr: float = 0.02, lbfgs_steps: int = 0,
                 relative_tolerance: float = 1e-4, gradient_tolerance: float = 1e-5,
                 patience: int = 3) -> dict[str, object]:
        """Optimize with iteration budgets and explicitly reported stopping criteria.

        ``adam_steps`` and ``lbfgs_steps`` are safety caps, not convergence
        claims. Relative decrease must be nonnegative for ``patience``
        consecutive accepted iterations. An objective increase resets it.
        """
        if min(adam_steps, lbfgs_steps) < 0 or adam_steps + lbfgs_steps < 1 or patience < 1:
            raise ValueError("positive total iteration budget and patience are required")
        if relative_tolerance <= 0 or gradient_tolerance <= 0 or adam_lr <= 0:
            raise ValueError("tolerances and learning rate must be positive")
        if initial_params is not None:
            value = np.asarray(initial_params).reshape(5, 6)
            if not np.isfinite(value).all():
                raise ValueError("initial_params must be finite")
            with torch.no_grad():
                self.params.copy_(_as_tensor(value, device=self.device, dtype=self.dtype))
        adam = torch.optim.Adam([self.params], lr=float(adam_lr))
        lbfgs = torch.optim.LBFGS([self.params], max_iter=1, tolerance_grad=0.0,
                                 tolerance_change=0.0, line_search_fn="strong_wolfe")
        evaluations = 0
        def closure() -> Tensor:
            nonlocal evaluations
            self.params.grad = None
            total, _ = self.loss()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("nonfinite registration objective")
            total.backward()
            if not bool(torch.isfinite(self.params.grad).all()):
                raise FloatingPointError("nonfinite registration gradient")
            evaluations += 1
            return total
        current = float(closure().detach())
        history = [current]
        diagnostics = []
        plateau = 0
        reason = "max_iterations"
        gradient = float(self.params.grad.detach().abs().max())
        iterations = 0
        if gradient <= gradient_tolerance:
            reason = "gradient_tolerance"
        else:
            for step in range(int(adam_steps) + int(lbfgs_steps)):
                if step < adam_steps:
                    adam.step()
                    phase = "adam"
                else:
                    lbfgs.step(closure)
                    phase = "lbfgs"
                updated = float(closure().detach())
                relative = (current - updated) / max(abs(current), 1e-12)
                plateau = plateau + 1 if 0 <= relative < relative_tolerance else 0
                gradient = float(self.params.grad.detach().abs().max())
                iterations = step + 1
                history.append(updated)
                diagnostics.append({"iteration": iterations, "phase": phase, "objective": updated,
                                    "relative_decrease": relative, "gradient_inf_norm": gradient})
                current = updated
                if gradient <= gradient_tolerance:
                    reason = "gradient_tolerance"
                    break
                if plateau >= patience:
                    reason = "relative_decrease"
                    break
        with torch.no_grad():
            final, parts = self.loss()
        return {"params": self.params.detach().cpu().numpy(), "loss": float(final.detach().cpu()),
                "image_loss": float(parts["image"].detach().cpu()), "motion_penalty": float(parts["motion"].detach().cpu()),
                "history": history, "device": str(self.device), "iterations": iterations,
                "stop_reason": reason, "converged": reason != "max_iterations",
                "reached_max_iterations": iterations == adam_steps + lbfgs_steps,
                "gradient_inf_norm": gradient, "gradient_condition_met": gradient <= gradient_tolerance,
                "relative_tolerance": relative_tolerance, "gradient_tolerance": gradient_tolerance,
                "patience": patience, "function_evaluations": evaluations, "iteration_diagnostics": diagnostics}
