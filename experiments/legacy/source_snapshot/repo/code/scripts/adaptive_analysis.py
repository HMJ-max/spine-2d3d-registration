"""Prespecified volume-level analysis for the adaptive-prior study.

Input rows in paired/proxy functions must already be matched by target and seed.
Repeated volume IDs are collapsed before mean/failure inference. No file loading
or outcome-dependent exclusions happen here; callers must audit completeness.
"""
from hashlib import sha256

import numpy as np
from scipy import stats

LAMBDA_GRID = (0.0, 0.003, 0.01, 0.03, 0.1)
FAILURE_THRESHOLDS_MM = (2.0, 3.0, 4.0, 5.0)
SPLIT_SALT = "adaptive_prior_v1_2026-10-24"
BOOTSTRAP_SEED = 20261024


def _hash(identifier, purpose):
    return sha256(f"{SPLIT_SALT}:{purpose}:{identifier}".encode()).hexdigest()


def deterministic_split(volume_ids, group_ids=None):
    """Return exactly 25 development/50 evaluation IDs and five grouped folds.

    group_ids optionally maps each volume to its verified patient identifier.
    If whole patient groups cannot make exactly 25 volumes, fail rather than
    silently split a patient. Fold entries are validation IDs; training is the
    development complement. All 75 volumes were previously exposed.
    """
    ids = list(volume_ids)
    if len(ids) != 75 or len(set(ids)) != 75 or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("Require 75 unique nonempty string volume IDs")
    if group_ids is not None and set(group_ids) != set(ids):
        raise ValueError("Patient mapping must cover exactly all 75 volumes")
    groups = {}
    for volume in ids:
        group = volume if group_ids is None else group_ids[volume]
        if not isinstance(group, str) or not group:
            raise ValueError("Group IDs must be nonempty strings")
        groups.setdefault(group, []).append(volume)
    ordered = sorted(groups, key=lambda x: (_hash(x, "split"), x))
    # First deterministic subset with exactly 25 volumes; never break a group.
    subsets = {0: []}
    for group in ordered:
        for size, selected in list(subsets.items()):
            total = size + len(groups[group])
            if total <= 25 and total not in subsets:
                subsets[total] = selected + [group]
    if 25 not in subsets:
        raise ValueError("Whole patient groups cannot form a 25/50 split")
    development_groups = subsets[25]
    if len(development_groups) < 5:
        raise ValueError("Development requires at least five independent groups")
    development = sorted(v for g in development_groups for v in groups[g])
    folds = [[] for _ in range(5)]
    for group in sorted(development_groups, key=lambda g: (-len(groups[g]), _hash(g, "fold"), g)):
        fold = min(range(5), key=lambda k: (len(folds[k]), k))
        folds[fold].extend(groups[group])
    return dict(development=development, evaluation=sorted(set(ids) - set(development)),
                development_folds=[sorted(fold) for fold in folds],
                all75_previously_exposed=True, salt=SPLIT_SALT,
                grouping="verified_patient" if group_ids is not None else "volume_independence_assumed")


def training_tau(proxies):
    """Median of training-fold edge proxies only, with a fixed numerical floor."""
    values = np.asarray(proxies, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Training proxies must be finite, nonnegative and nonempty")
    return max(float(np.median(values)), 1e-12)


def select_lambda(clean_rms, combined_rms, lambdas=LAMBDA_GRID):
    """One-SE selection using OUT-OF-FOLD risks, one equally weighted row/volume.

    clean_rms is [volume, lambda]; combined_rms is [volume, lambda, seed].
    Adaptive predictions for a validation fold must use tau fitted exclusively
    on its training complement. Refit tau on all development data afterwards.
    """
    clean, combined = np.asarray(clean_rms, float), np.asarray(combined_rms, float)
    grid = np.asarray(lambdas, float)
    if tuple(grid) != LAMBDA_GRID:
        raise ValueError("Use the locked lambda grid")
    if (clean.ndim != 2 or clean.shape[0] < 2 or clean.shape[1] != len(grid)
            or combined.ndim != 3 or combined.shape[:2] != clean.shape or combined.shape[2] != 5):
        raise ValueError("Expected clean [volume,5] and combined [volume,5,5 seeds]")
    if not np.isfinite(clean).all() or not np.isfinite(combined).all() or np.any(clean < 0) or np.any(combined < 0):
        raise ValueError("RMS losses must be finite and nonnegative")
    risks = 0.5 * clean + 0.5 * combined.mean(axis=2)
    means, se = risks.mean(axis=0), risks.std(axis=0, ddof=1) / np.sqrt(len(risks))
    best = int(np.argmin(means))
    cutoff = means[best] + se[best]
    selected = int(np.flatnonzero(means <= cutoff)[0])
    return dict(selected_lambda=float(grid[selected]), minimum_risk_lambda=float(grid[best]),
                one_se_cutoff=float(cutoff), mean_risk=means.tolist(), standard_error=se.tolist(),
                n_volumes=len(risks), lambdas=grid.tolist())


def _rows(values, volume_ids):
    values = np.asarray(values, dtype=float)
    ids = list(volume_ids)
    if values.ndim != 1 or len(values) != len(ids) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Require one finite value per row and a matching volume ID")
    if not all(isinstance(x, str) and x for x in ids):
        raise ValueError("Volume IDs must be nonempty strings")
    unique, inverse = np.unique(ids, return_inverse=True)
    return values, unique, inverse


def _volume_means(values, volume_ids):
    values, unique, inverse = _rows(values, volume_ids)
    return unique, np.bincount(inverse, weights=values) / np.bincount(inverse)


def _bootstrap_means(values, bootstrap_samples, seed):
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("Require at least 100 bootstrap draws (10000 for study reporting)")
    rng = np.random.default_rng(seed)
    return np.mean(values[rng.integers(0, len(values), size=(bootstrap_samples, len(values)))], axis=1)


def paired_comparison(first, second, volume_ids, *, bootstrap_samples=10000, seed=BOOTSTRAP_SEED):
    """Mean first-minus-second difference; lower is better, seeds average within CT.

    Primary p is a lower-tail null-centered paired bootstrap test. ci95 is the
    percentile interval; basic_ci95 inverts the centered test (Monte Carlo
    discreteness can differ by one draw). t-test is sensitivity evidence only.
    """
    a, unique, inverse = _rows(first, volume_ids)
    b, _, _ = _rows(second, volume_ids)
    if len(unique) < 2:
        raise ValueError("At least two independent volumes are required")
    differences = np.bincount(inverse, weights=a - b) / np.bincount(inverse)
    mean = float(differences.mean())
    draws = _bootstrap_means(differences, bootstrap_samples, seed)
    ci = np.quantile(draws, [.025, .975])
    p = float((1 + np.count_nonzero(draws - mean <= mean)) / (bootstrap_samples + 1))
    identical = bool(np.all(differences == 0))
    if identical:
        p = 1.0
    variance = float(np.var(differences, ddof=1))
    constant = np.allclose(differences, differences[0], rtol=1e-12, atol=1e-15)
    t_p = float(stats.ttest_1samp(differences, 0, alternative="less").pvalue) if not constant else (1.0 if identical else None)
    return dict(n_volumes=len(unique), n_rows=len(a), difference=mean, ci95=ci.tolist(),
                basic_ci95=[float(2 * mean - ci[1]), float(2 * mean - ci[0])],
                p_one_sided=p, alpha=.025, superiority=bool(mean < 0 and p <= .025),
                t_p_one_sided=t_p, bootstrap_samples=bootstrap_samples, bootstrap_seed=seed,
                zero_empirical_variance=variance == 0, numerically_constant_difference=bool(constant),
                volume_ids=unique.tolist(),
                volume_differences=differences.tolist())


def joint_superiority(rms_first, rms_second, failure_first, failure_second, volume_ids, **bootstrap_options):
    """Intersection-union test: both prespecified endpoints must reject at .025."""
    for values in (failure_first, failure_second):
        if not np.isin(np.asarray(values), [0, 1]).all():
            raise ValueError("Failure rows must be binary, before volume seed averaging")
    rms = paired_comparison(rms_first, rms_second, volume_ids, **bootstrap_options)
    failure = paired_comparison(failure_first, failure_second, volume_ids, **bootstrap_options)
    return dict(rms=rms, failure_frequency=failure, joint_superiority=rms["superiority"] and failure["superiority"],
                joint_p_one_sided=max(rms["p_one_sided"], failure["p_one_sided"]), family_alpha=.025)


def _auroc(scores, labels, weights):
    order = np.argsort(scores, kind="stable")
    scores, labels, weights = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positive = np.add.reduceat(weights * labels, starts)
    negative = np.add.reduceat(weights * (1 - labels), starts)
    denominator = positive.sum() * negative.sum()
    if denominator == 0:
        return np.nan
    return float(np.sum(positive * (np.cumsum(negative) - .5 * negative)) / denominator)


def proxy_gate(scores, failures, volume_ids, *, bootstrap_samples=10000, seed=BOOTSTRAP_SEED):
    """Equal-volume weighted AUROC with whole-CT bootstrap; larger proxy=worse.

    Scores are per-target maximum edge dispersion, labels are B's >3mm failure.
    Every volume contributes total weight one, even if it has repeated targets.
    """
    scores, unique, inverse = _rows(scores, volume_ids)
    labels, _, _ = _rows(failures, volume_ids)
    if not np.isin(labels, [0, 1]).all() or np.any(scores < 0):
        raise ValueError("Require binary failures and nonnegative proxy scores")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("Require at least 100 bootstrap draws")
    counts = np.bincount(inverse)
    weights = 1 / counts[inverse]
    auc = _auroc(scores, labels, weights)
    positive_volumes = len(np.unique(inverse[labels == 1]))
    negative_volumes = len(np.unique(inverse[labels == 0]))
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap_samples):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        value = _auroc(scores, labels, weights * multiplicity[inverse])
        if np.isfinite(value):
            draws.append(value)
    ci = np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]
    sufficient = positive_volumes >= 5 and negative_volumes >= 5 and len(draws) >= .95 * bootstrap_samples
    return dict(auroc=float(auc) if np.isfinite(auc) else None, ci95=ci,
                positive_volumes=positive_volumes, negative_volumes=negative_volumes,
                n_volumes=len(unique), valid_bootstrap_draws=len(draws), bootstrap_samples=bootstrap_samples,
                bootstrap_seed=seed, evidence_sufficient=bool(sufficient),
                gate_passed=bool(sufficient and auc >= .70 and ci[0] > .50))


def summarize_outcomes(rms, centroid_errors, volume_ids, *, bootstrap_samples=10000, seed=BOOTSTRAP_SEED):
    """RMS summaries across seed-averaged volumes; centroid rows are [row,5].

    Failure is any of the five centroid errors strictly greater than threshold.
    Median/p95 describe the distribution of volume seed means, not seed tails.
    """
    values, unique, _ = _rows(rms, volume_ids)
    errors = np.asarray(centroid_errors, float)
    if errors.shape != (len(values), 5) or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(values < 0):
        raise ValueError("Require nonnegative RMS and five finite centroid errors per row")
    _, means = _volume_means(values, volume_ids)
    draws = _bootstrap_means(means, bootstrap_samples, seed)
    failures = {}
    for threshold in FAILURE_THRESHOLDS_MM:
        _, frequencies = _volume_means(np.any(errors > threshold, axis=1), volume_ids)
        intervals = np.quantile(_bootstrap_means(frequencies, bootstrap_samples, seed), [.025, .975])
        failures[str(int(threshold))] = dict(mean_frequency=float(frequencies.mean()), ci95=intervals.tolist(),
                                             volume_frequencies=frequencies.tolist())
    return dict(n_volumes=len(unique), n_rows=len(values), mean_rms=float(means.mean()),
                mean_rms_ci95=np.quantile(draws, [.025, .975]).tolist(), median_rms=float(np.median(means)),
                p95_rms=float(np.quantile(means, .95)), volume_ids=unique.tolist(), volume_rms=means.tolist(),
                failures_by_threshold_mm=failures)
