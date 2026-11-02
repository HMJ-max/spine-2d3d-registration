"""Auditable optimizers for the corrected, image-based registration benchmark.

All public poses contain axis-angle rotation in radians followed by translation
in millimetres. Objectives receive these physical coordinates, never normalized
optimizer coordinates. No optimizer accepts a ground-truth pose.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
import torch
from torch import Tensor

from .torch_registration import so3_exp


def kd_partition_starts(count: int, bounds: np.ndarray) -> list[np.ndarray]:
    """Centers of a deterministic k-d partition, plus the zero-motion center.

    Split the largest remaining cell, cycling split axes across all splits.
    This explores all six coordinates with eight cells instead of splitting
    only the first three coordinates in a balanced depth-three tree.
    """
    bounds = np.asarray(bounds, dtype=float)
    if count < 2 or bounds.shape != (6,) or not np.all(np.isfinite(bounds) & (bounds > 0)):
        raise ValueError("At least two cells and six positive finite half-widths are required")
    cells = [(np.full(6, -1.0), np.ones(6))]
    for split in range(count - 1):
        index = int(np.argmax([np.prod(upper - lower) for lower, upper in cells]))
        lower, upper = cells.pop(index)
        axis = split % 6
        midpoint = (lower[axis] + upper[axis]) * 0.5
        first_upper, second_lower = upper.copy(), lower.copy()
        first_upper[axis] = second_lower[axis] = midpoint
        cells.extend([(lower.copy(), first_upper), (second_lower, upper.copy())])
    return [np.zeros(6), *[(lower + upper) * 0.5 * bounds for lower, upper in cells]]


def optimize_powell(
    objective_factory: Callable[[int], Callable[[np.ndarray], float]],
    initial_pose: np.ndarray | None = None,
    *,
    levels: Sequence[int] = (64, 128, 256),
    starts: int = 8,
    rotation_bound_deg: float = 3.0,
    translation_bound_mm: float = 5.0,
    max_iterations: int = 500,
    xtol: float = 1e-4,
    ftol: float = 1e-5,
) -> dict:
    """Coarse k-d multi-start Powell followed by single-winner refinement.

    ``starts`` is the number of partition centers; one identity/initial-pose
    start is added. SciPy xtol controls line searches; an explicit callback
    also checks normalized outer-step size. Both stop reasons are recorded.
    """
    initial = np.zeros(6) if initial_pose is None else np.asarray(initial_pose, dtype=float)
    levels = tuple(int(level) for level in levels)
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_pose must contain six finite physical coordinates")
    if not levels or any(level < 1 for level in levels) or max_iterations < 1 or xtol <= 0 or ftol <= 0:
        raise ValueError("Positive levels, tolerances and maximum iterations are required")
    scale = np.array([math.radians(0.01)] * 3 + [1.0] * 3)
    half_width = np.array([math.radians(rotation_bound_deg)] * 3 + [translation_bound_mm] * 3)
    start_offsets = kd_partition_starts(starts, half_width)
    q_lower, q_upper = (initial - half_width) / scale, (initial + half_width) / scale
    winner = initial.copy()
    history, diagnostics = [], []
    total_evaluations, total_iterations = 0, 0
    for level_index, level in enumerate(levels):
        objective = objective_factory(level)
        cache: dict[tuple, float] = {}

        def evaluate(q):
            key = tuple(np.asarray(q, dtype=float))
            if key not in cache:
                loss = float(objective(np.asarray(q) * scale))
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite Powell objective at level {level}")
                cache[key] = loss
            return cache[key]

        candidates = [initial + offset for offset in start_offsets] if level_index == 0 else [winner]
        winner_value = math.inf
        for start_index, candidate in enumerate(candidates):
            before_count = len(cache)
            q0 = candidate / scale
            initial_value = evaluate(q0)
            accepted = []
            previous = {"q": q0.copy(), "loss": initial_value}

            def callback(q):
                value = evaluate(q)
                step = float(np.max(np.abs(q - previous["q"])))
                record = {"level": level, "start": start_index, "iteration": len(accepted) + 1,
                          "loss": value, "pose": (q * scale).tolist(),
                          "normalized_step_inf": step, "objective_decrease": previous["loss"] - value}
                accepted.append(record)
                previous.update(q=q.copy(), loss=value)
                if step < xtol:
                    raise StopIteration

            result = minimize(evaluate, q0, method="Powell", bounds=list(zip(q_lower, q_upper)),
                              callback=callback,
                              options={"xtol": xtol, "ftol": ftol, "maxiter": max_iterations})
            # A numerical or resource-limit exit must not discard a better
            # already evaluated candidate, including the zero-motion start.
            trial_points = [(initial_value, q0), (float(result.fun), result.x)]
            trial_points.extend((row["loss"], np.asarray(row["pose"]) / scale) for row in accepted)
            value, selected_q = min(trial_points, key=lambda item: item[0])
            if value < winner_value:
                winner_value, winner = value, selected_q * scale
            status = int(result.status)
            reason = {0: "metric_tolerance", 1: "max_function_evaluations", 2: "max_iterations",
                      3: "nonfinite", 4: "outside_bounds", 99: "step_tolerance"}.get(status, "optimizer_failure")
            diagnostics.append({"level": level, "start": start_index, "initial_pose": candidate.tolist(),
                                "iterations": int(result.nit), "function_calls": int(result.nfev),
                                "uncached_evaluations": len(cache) - before_count,
                                "initial_loss": initial_value, "final_loss": float(result.fun),
                                "selected_loss": value, "stop_reason": reason,
                                "converged": bool(result.success) or status == 99, "status": status,
                                "message": str(result.message), "hit_max_iterations": int(result.nit) >= max_iterations,
                                "last_normalized_step_inf": accepted[-1]["normalized_step_inf"] if accepted else None,
                                "triggers": {"step_tolerance": bool(accepted and accepted[-1]["normalized_step_inf"] < xtol),
                                             "metric_tolerance": status == 0,
                                             "max_iterations": status == 2,
                                             "max_function_evaluations": status == 1},
                                "xtol_role": "line_search_accuracy_plus_explicit_normalized_outer_step_callback"})
            history.extend(accepted)
            total_iterations += int(result.nit)
        total_evaluations += len(cache)
    return {"pose": winner.tolist(), "loss": winner_value, "iterations": total_iterations,
            "evaluations": total_evaluations, "history": history, "diagnostics": diagnostics,
            "stop_reason": diagnostics[-1]["stop_reason"],
            "converged": diagnostics[-1]["converged"],
            "hit_max_iterations": any(row["hit_max_iterations"] for row in diagnostics),
            "config": {"levels": list(levels), "partition_starts": starts, "start_count": starts + 1,
                       "rotation_bound_deg": rotation_bound_deg, "translation_bound_mm": translation_bound_mm,
                       "rotation_radians_per_search_unit": scale[0], "translation_mm_per_search_unit": 1.0,
                       "xtol": xtol, "ftol": ftol, "max_iterations_per_start_and_level": max_iterations}}


def _pose_increment(previous: Tensor, current: Tensor) -> tuple[float, float]:
    old_r, new_r = so3_exp(previous[..., :3]), so3_exp(current[..., :3])
    delta = new_r @ old_r.transpose(-1, -2)
    skew = torch.stack((delta[..., 2, 1] - delta[..., 1, 2],
                        delta[..., 0, 2] - delta[..., 2, 0],
                        delta[..., 1, 0] - delta[..., 0, 1]), dim=-1) * 0.5
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = (torch.diagonal(delta, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    angle = torch.atan2(sine, cosine.clamp(-1, 1)).abs().max()
    distance = torch.linalg.vector_norm(current[..., 3:] - previous[..., 3:], dim=-1).max()
    return float(angle), float(distance)


def optimize_torch(
    objective: Callable[[Tensor], Tensor],
    initial_pose: Tensor,
    *,
    mode: str,
    max_iterations: int = 500,
    adam_warmup: int = 30,
    adam_lr: float = 0.2,
    relative_tolerance: float = 1e-4,
    relative_patience: int = 5,
    gradient_tolerance: float = 1e-5,
    absolute_loss_tolerance: float = 1e-4,
    rotation_tolerance_rad: float = 1e-4,
    translation_tolerance_mm: float = 1e-4,
) -> dict:
    """Image-based independent or jointly regularized pose optimization.

    The independent mode applies ICP-style stopping tests to an intensity
    objective; it is not point-cloud ICP and never claims correspondence RMS.
    The joint mode requires consecutive strictly positive small decreases.
    Unchanged iterates with a large gradient are reported as stagnation.
    """
    if mode not in {"independent", "joint"}:
        raise ValueError("mode must be independent or joint")
    if initial_pose.shape[-1] != 6 or not initial_pose.is_floating_point() or not torch.isfinite(initial_pose).all():
        raise ValueError("initial_pose must be a finite floating tensor with final dimension six")
    if max_iterations < 1 or adam_warmup < 0 or relative_patience < 1:
        raise ValueError("Invalid iteration limits")
    if min(adam_lr, relative_tolerance, gradient_tolerance, absolute_loss_tolerance,
           rotation_tolerance_rad, translation_tolerance_mm) <= 0:
        raise ValueError("Learning rate and all stopping tolerances must be positive")
    scale = initial_pose.new_tensor([math.pi / 180.0] * 3 + [1.0] * 3)
    q = torch.nn.Parameter(initial_pose.detach().clone() / scale)
    adam = torch.optim.Adam([q], lr=adam_lr)
    lbfgs = torch.optim.LBFGS([q], lr=1.0, max_iter=1, max_eval=25,
                            tolerance_grad=0.0, tolerance_change=0.0,
                            history_size=20, line_search_fn="strong_wolfe")
    evaluations = 0
    cached_q, cached_loss, cached_gradient = None, None, None

    def closure():
        nonlocal evaluations, cached_q, cached_loss, cached_gradient
        if cached_q is not None and torch.equal(q.detach(), cached_q):
            q.grad = cached_gradient.clone()
            return cached_loss
        q.grad = None
        loss = objective(q * scale)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Objective must return a finite scalar tensor")
        loss.backward()
        if q.grad is None or not torch.isfinite(q.grad).all():
            raise FloatingPointError("Nonfinite or disconnected pose gradient")
        evaluations += 1
        cached_q, cached_loss, cached_gradient = q.detach().clone(), loss.detach(), q.grad.detach().clone()
        return cached_loss

    value = float(closure())
    best_q, best_value = q.detach().clone(), value
    history = []
    small_decreases, unchanged_count = 0, 0
    reason = "max_iterations"
    gradient_norm = float(torch.linalg.vector_norm(q.grad))
    triggers = {"absolute_loss": mode == "independent" and value <= absolute_loss_tolerance,
                "gradient": mode == "joint" and gradient_norm <= gradient_tolerance,
                "pose_increment": False, "relative_decrease": False, "max_iterations": False}
    if triggers["absolute_loss"] or triggers["gradient"]:
        reason = "absolute_loss" if triggers["absolute_loss"] else "gradient"
    else:
        for iteration in range(1, max_iterations + 1):
            if iteration == adam_warmup + 1:
                with torch.no_grad():
                    q.copy_(best_q)
                value = float(closure())
            previous = (q.detach() * scale).clone()
            old_value = value
            phase = "adam" if iteration <= adam_warmup else "lbfgs_strong_wolfe"
            previous_internal_iterations = int(lbfgs.state.get(q, {}).get("n_iter", 0))
            if phase == "adam":
                adam.step()
            else:
                lbfgs.step(closure)
            value = float(closure())
            physical_pose = q.detach() * scale
            delta_rotation, delta_translation = _pose_increment(previous, physical_pose)
            gradient_norm = float(torch.linalg.vector_norm(q.grad))
            decrease = old_value - value
            relative_decrease = decrease / max(abs(old_value), 1e-12)
            unchanged = torch.equal(previous, physical_pose)
            unchanged_count = unchanged_count + 1 if unchanged else 0
            if phase != "adam" and 0.0 < relative_decrease < relative_tolerance and not unchanged:
                small_decreases += 1
            else:
                small_decreases = 0
            if value < best_value:
                best_q, best_value = q.detach().clone(), value
            triggers = {
                "absolute_loss": mode == "independent" and value <= absolute_loss_tolerance,
                "gradient": mode == "joint" and gradient_norm <= gradient_tolerance,
                "pose_increment": mode == "independent" and phase != "adam" and
                                  delta_rotation < rotation_tolerance_rad and delta_translation < translation_tolerance_mm,
                "relative_decrease": mode == "joint" and small_decreases >= relative_patience,
                "max_iterations": iteration >= max_iterations,
            }
            history.append({"iteration": iteration, "phase": phase, "loss": value,
                            "optimizer_internal_iterations": 1 if phase == "adam" else
                                int(lbfgs.state.get(q, {}).get("n_iter", 0)) - previous_internal_iterations,
                            "accepted_pose_update": not unchanged,
                            "objective_decrease": decrease, "relative_decrease": relative_decrease,
                            "gradient_norm_scaled_coordinates": gradient_norm,
                            "increment_rotation_rad": delta_rotation, "increment_translation_mm": delta_translation,
                            "pose": physical_pose.cpu().tolist(), "nonincreasing": decrease >= 0.0,
                            "unchanged": unchanged, "triggers": triggers.copy()})
            triggered = next((key for key, active in triggers.items() if active), None)
            if triggered:
                reason = triggered
                break
            if phase != "adam" and unchanged_count >= 3:
                reason = "numerical_stagnation"
                break
    # Return the best image/prior objective encountered, with the actual last
    # iterate diagnostics retained so rollback cannot masquerade as convergence.
    last_pose = (q.detach() * scale).clone()
    best_pose = best_q * scale
    rollback = not torch.equal(last_pose, best_pose)
    return {"pose": best_pose, "loss": best_value, "iterations": len(history),
            "accepted_pose_updates": sum(row["accepted_pose_update"] for row in history),
            "optimizer_internal_iterations": sum(row["optimizer_internal_iterations"] for row in history),
            "evaluations": evaluations, "history": history,
            "diagnostics": {"stop_reason": reason, "triggers": triggers,
                            "converged": reason not in {"max_iterations", "numerical_stagnation"},
                            "hit_max_iterations": triggers["max_iterations"],
                            "final_iterate_loss": value, "returned_best_iterate": rollback,
                            "final_gradient_norm_scaled_coordinates": gradient_norm,
                            "numerical_stagnation": unchanged_count > 0 and gradient_norm > gradient_tolerance,
                            "relative_decrease_streak": small_decreases},
            "stop_reason": reason,
            "converged": reason not in {"max_iterations", "numerical_stagnation"},
            "hit_max_iterations": triggers["max_iterations"],
            "config": {"mode": mode, "max_iterations": max_iterations,
                       "iteration_definition": "outer_update_attempt; actual_nonzero_updates_and_internal_iterations_also_recorded",
                       "adam_warmup_max_iterations": adam_warmup, "adam_lr": adam_lr,
                       "lbfgs_line_search": "strong_wolfe", "relative_tolerance": relative_tolerance,
                       "relative_patience": relative_patience, "gradient_tolerance": gradient_tolerance,
                       "gradient_coordinates": "rotation_degrees_translation_mm",
                       "absolute_loss_tolerance": absolute_loss_tolerance,
                       "rotation_tolerance_rad": rotation_tolerance_rad,
                       "translation_tolerance_mm": translation_tolerance_mm}}


def optimize_independent_blocks(
    objective: Callable[[Tensor], Tensor],
    initial_pose: Tensor,
    *,
    block_objective: Callable[[int, Tensor], Callable[[Tensor], Tensor]] | None = None,
    max_iterations: int = 500,
    loss_threshold: float = 1e-6,
    rotation_tolerance_rad: float = 1e-4,
    translation_tolerance_mm: float = 1e-4,
    block_steps: int = 5,
) -> dict:
    """Unregularized intensity registration using independent pose blocks.

    Each vertex observes the same full composite target. Its neighbors are
    frozen only during a block visit and are revisited in subsequent sweeps.
    ``block_steps`` is a scheduling budget, not a convergence criterion.
    The optional factory can cache the frozen neighbors' projected images.
    """
    if initial_pose.ndim != 2 or initial_pose.shape[1] != 6 or initial_pose.shape[0] < 1:
        raise ValueError("Expected an N-by-6 physical pose tensor")
    if max_iterations < 1 or block_steps < 1 or loss_threshold <= 0:
        raise ValueError("Positive iteration budgets and loss threshold are required")
    poses = initial_pose.detach().clone()
    count = len(poses)
    iterations = [0] * count
    visits: list[list[dict]] = [[] for _ in range(count)]
    history = []
    evaluations = 0

    def full_loss():
        nonlocal evaluations
        with torch.no_grad():
            value = float(objective(poses))
        evaluations += 1
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite composite objective")
        return value

    value = full_loss()
    reason = "absolute_loss" if value <= loss_threshold else "max_iterations"
    last_increments = [(None, None)] * count
    while value > loss_threshold and max(iterations) < max_iterations:
        sweep_start = poses.clone()
        for index in range(count):
            frozen = poses.clone()
            if block_objective is None:
                def current_objective(pose, index=index, frozen=frozen):
                    return objective(torch.cat((frozen[:index], pose.unsqueeze(0), frozen[index + 1:]), dim=0))
            else:
                current_objective = block_objective(index, frozen)
            result = optimize_torch(
                current_objective, poses[index], mode="independent", adam_warmup=0,
                max_iterations=min(block_steps, max_iterations - iterations[index]),
                absolute_loss_tolerance=loss_threshold,
                rotation_tolerance_rad=rotation_tolerance_rad,
                translation_tolerance_mm=translation_tolerance_mm,
            )
            poses[index] = result["pose"]
            iterations[index] += result["iterations"]
            evaluations += result["evaluations"]
            visits[index].append({"sweep": len(history) + 1, "iterations": result["iterations"],
                                  "loss": result["loss"], "stop_reason": result["stop_reason"],
                                  "returned_best_iterate": result["diagnostics"]["returned_best_iterate"],
                                  "numerical_stagnation": result["diagnostics"]["numerical_stagnation"],
                                  "inner_budget_exhausted": result["hit_max_iterations"],
                                  "history": result["history"]})
        value = full_loss()
        last_increments = [_pose_increment(sweep_start[index], poses[index]) for index in range(count)]
        small_increment = all(rotation < rotation_tolerance_rad and translation < translation_tolerance_mm
                              for rotation, translation in last_increments)
        triggers = {"absolute_loss": value <= loss_threshold,
                    "pose_increment": small_increment,
                    "max_iterations": max(iterations) >= max_iterations}
        history.append({"sweep": len(history) + 1, "loss": value, "pose": poses.cpu().tolist(),
                        "iterations_per_vertebra": iterations.copy(),
                        "increment_rotation_rad": [pair[0] for pair in last_increments],
                        "increment_translation_mm": [pair[1] for pair in last_increments],
                        "triggers": triggers})
        triggered = next((key for key, active in triggers.items() if active), None)
        if triggered:
            reason = triggered
            break
    hit_max = max(iterations) >= max_iterations
    vertebra_diagnostics = []
    for index in range(count):
        rotation, translation = last_increments[index]
        vertebra_diagnostics.append({
            "index": index, "iterations": iterations[index], "loss": value,
            "loss_scope": "full_observed_composite_AP_LAT",
            "stop_reason": "shared_budget_exhausted" if reason == "max_iterations" and iterations[index] < max_iterations else reason,
            "hit_max_iterations": iterations[index] >= max_iterations,
            "increment_rotation_rad": rotation, "increment_translation_mm": translation,
            "triggers": {"shared_budget_exhausted": reason == "max_iterations" and iterations[index] < max_iterations,
                         "absolute_loss": value <= loss_threshold,
                         "pose_increment": rotation is not None and rotation < rotation_tolerance_rad and
                                           translation < translation_tolerance_mm,
                         "max_iterations": iterations[index] >= max_iterations},
            "visits": visits[index],
        })
    return {"pose": poses, "loss": value, "iterations": sum(iterations),
            "iterations_per_vertebra": iterations, "sweeps": len(history),
            "evaluations": evaluations, "history": history, "diagnostics": vertebra_diagnostics,
            "stop_reason": reason, "hit_max_iterations": hit_max,
            "converged": reason in {"absolute_loss", "pose_increment"},
            "config": {"optimizer": "block_coordinate_LBFGS_strong_wolfe", "icp": False,
                       "cross_vertebra_regularization": False, "max_iterations_per_vertebra": max_iterations,
                       "block_steps_scheduling_budget": block_steps, "loss_threshold": loss_threshold,
                       "rotation_tolerance_rad": rotation_tolerance_rad,
                       "translation_tolerance_mm": translation_tolerance_mm}}
