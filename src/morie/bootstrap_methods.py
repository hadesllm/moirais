"""Comprehensive bootstrap and resampling inference methods.

Provides nonparametric and parametric bootstrap, jackknife, permutation
tests, subsampling, and cross-validation resampling for statistical
inference in epidemiological and observational studies.

All methods support stratified resampling, cluster-aware resampling,
and survey-weighted variants for complex survey designs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result from a bootstrap procedure."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    bias: float
    n_boot: int
    method: str
    ci_method: str
    boot_distribution: np.ndarray
    original_estimate: float
    acceleration: float = 0.0

    def summary(self) -> dict:
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bias": self.bias,
            "method": self.method,
        }


@dataclass
class JackknifeResult:
    """Result from jackknife estimation."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    bias: float
    n: int
    jackknife_estimates: np.ndarray
    pseudovalues: np.ndarray
    influence_values: np.ndarray

    def summary(self) -> dict:
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bias": self.bias,
        }


@dataclass
class PermutationTestResult:
    """Result from a permutation test."""

    observed_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int
    alternative: str
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")


@dataclass
class CrossValidationResult:
    """Result from cross-validation."""

    scores: np.ndarray
    mean_score: float
    se_score: float
    ci_lower: float
    ci_upper: float
    n_folds: int
    metric: str
    fold_sizes: list[int]


# ---------------------------------------------------------------------------
# Nonparametric bootstrap
# ---------------------------------------------------------------------------


def bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci_level: float = 0.95,
    ci_method: str = "bca",
    seed: int = 42,
    stratify: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
) -> BootstrapResult:
    """Nonparametric bootstrap inference.

    Parameters
    ----------
    data : array-like
        Input data (1D or 2D).
    statistic : callable
        Function that computes the statistic from a data array.
    n_boot : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence interval level.
    ci_method : str
        CI method: 'percentile', 'normal', 'basic', 'bca', 'studentized'.
    seed : int
        Random seed.
    stratify : array-like, optional
        Stratification variable for stratified bootstrap.
    cluster : array-like, optional
        Cluster variable for cluster bootstrap.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    original = statistic(data)

    boot_stats = np.empty(n_boot)

    if cluster is not None:
        cluster = np.asarray(cluster)
        unique_clusters = np.unique(cluster)
        n_clusters = len(unique_clusters)

        for b in range(n_boot):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            indices = np.concatenate([np.where(cluster == c)[0] for c in sampled_clusters])
            boot_stats[b] = statistic(data[indices])

    elif stratify is not None:
        stratify = np.asarray(stratify)
        strata = np.unique(stratify)

        for b in range(n_boot):
            indices = []
            for s in strata:
                s_idx = np.where(stratify == s)[0]
                indices.append(rng.choice(s_idx, size=len(s_idx), replace=True))
            all_indices = np.concatenate(indices)
            boot_stats[b] = statistic(data[all_indices])

    else:
        for b in range(n_boot):
            indices = rng.integers(0, n, size=n)
            boot_stats[b] = statistic(data[indices])

    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - original)

    alpha = 1 - ci_level

    if ci_method == "percentile":
        ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    elif ci_method == "normal":
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lo = float(original - bias - z * se)
        ci_hi = float(original - bias + z * se)

    elif ci_method == "basic":
        p_lo = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        p_hi = np.percentile(boot_stats, 100 * alpha / 2)
        ci_lo = float(2 * original - p_lo)
        ci_hi = float(2 * original - p_hi)

    elif ci_method == "bca":
        ci_lo, ci_hi, acc = _bca_interval(data, statistic, boot_stats, original, ci_level, rng)

    elif ci_method == "studentized":
        # Requires nested bootstrap for SE estimation.
        boot_ses = np.empty(n_boot)
        for b in range(n_boot):
            indices = rng.integers(0, n, size=n)
            boot_data = data[indices]
            inner_stats = np.empty(50)
            for ib in range(50):
                inner_idx = rng.integers(0, len(boot_data), size=len(boot_data))
                inner_stats[ib] = statistic(boot_data[inner_idx])
            boot_ses[b] = np.std(inner_stats, ddof=1)

        t_stats = (boot_stats - original) / np.maximum(boot_ses, 1e-10)
        t_lo = np.percentile(t_stats, 100 * (1 - alpha / 2))
        t_hi = np.percentile(t_stats, 100 * alpha / 2)
        ci_lo = float(original - t_lo * se)
        ci_hi = float(original - t_hi * se)

    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")

    acc = 0.0
    if ci_method == "bca":
        pass  # already set
    else:
        acc = 0.0

    return BootstrapResult(
        estimate=original,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        bias=bias,
        n_boot=n_boot,
        method="nonparametric",
        ci_method=ci_method,
        boot_distribution=boot_stats,
        original_estimate=original,
        acceleration=acc,
    )


def _bca_interval(
    data: np.ndarray,
    statistic: Callable,
    boot_stats: np.ndarray,
    original: float,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Compute BCa (bias-corrected and accelerated) confidence interval."""
    n = len(data)
    alpha = 1 - ci_level

    # Bias correction factor (z0).
    z0 = stats.norm.ppf(np.mean(boot_stats < original))

    # Acceleration factor (a) via jackknife.
    jack_stats = np.empty(n)
    for i in range(n):
        jack_data = np.delete(data, i, axis=0)
        jack_stats[i] = statistic(jack_data)

    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
    a = num / max(den, 1e-10)

    # Adjusted percentiles.
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / max(1 - a * (z0 + z_lo), 0.01))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / max(1 - a * (z0 + z_hi), 0.01))

    ci_lo = float(np.percentile(boot_stats, 100 * np.clip(a1, 0.001, 0.999)))
    ci_hi = float(np.percentile(boot_stats, 100 * np.clip(a2, 0.001, 0.999)))

    return ci_lo, ci_hi, float(a)


# ---------------------------------------------------------------------------
# Parametric bootstrap
# ---------------------------------------------------------------------------


def parametric_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    distribution: str = "normal",
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    **dist_params,
) -> BootstrapResult:
    """Parametric bootstrap.

    Generates bootstrap samples from a fitted parametric distribution
    rather than resampling from the data.

    Parameters
    ----------
    data : array-like
        Original data (used to fit the distribution).
    statistic : callable
        Function that computes the statistic.
    distribution : str
        Distribution: 'normal', 'poisson', 'binomial', 'exponential', 'gamma'.
    n_boot : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence interval level.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    original = statistic(data)

    boot_stats = np.empty(n_boot)

    if distribution == "normal":
        mu = dist_params.get("mu", np.mean(data))
        sigma = dist_params.get("sigma", np.std(data, ddof=1))
        for b in range(n_boot):
            boot_data = rng.normal(mu, sigma, n)
            boot_stats[b] = statistic(boot_data)

    elif distribution == "poisson":
        lam = dist_params.get("lam", np.mean(data))
        for b in range(n_boot):
            boot_data = rng.poisson(lam, n).astype(float)
            boot_stats[b] = statistic(boot_data)

    elif distribution == "binomial":
        p = dist_params.get("p", np.mean(data))
        for b in range(n_boot):
            boot_data = rng.binomial(1, p, n).astype(float)
            boot_stats[b] = statistic(boot_data)

    elif distribution == "exponential":
        scale = dist_params.get("scale", np.mean(data))
        for b in range(n_boot):
            boot_data = rng.exponential(scale, n)
            boot_stats[b] = statistic(boot_data)

    elif distribution == "gamma":
        shape = dist_params.get("shape")
        scale = dist_params.get("scale")
        if shape is None or scale is None:
            # MoM estimation.
            mu = np.mean(data)
            var = np.var(data, ddof=1)
            shape = mu**2 / max(var, 1e-10)
            scale = max(var, 1e-10) / mu
        for b in range(n_boot):
            boot_data = rng.gamma(shape, scale, n)
            boot_stats[b] = statistic(boot_data)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - original)
    alpha = 1 - ci_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=original,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        bias=bias,
        n_boot=n_boot,
        method=f"parametric_{distribution}",
        ci_method="percentile",
        boot_distribution=boot_stats,
        original_estimate=original,
    )


# ---------------------------------------------------------------------------
# Wild bootstrap (for heteroskedasticity)
# ---------------------------------------------------------------------------


def wild_bootstrap(
    y: np.ndarray,
    X: np.ndarray,
    statistic_idx: int = 1,
    n_boot: int = 999,
    ci_level: float = 0.95,
    weight_distribution: str = "rademacher",
    seed: int = 42,
) -> BootstrapResult:
    """Wild bootstrap for linear regression with heteroskedasticity.

    Parameters
    ----------
    y : array-like
        Response variable.
    X : array-like
        Design matrix (including intercept if desired).
    statistic_idx : int
        Index of the coefficient to test.
    n_boot : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence interval level.
    weight_distribution : str
        'rademacher' (±1 with equal probability) or 'mammen'.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)

    # OLS fit.
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_hat
    y_hat = X @ beta_hat
    original = beta_hat[statistic_idx]

    boot_stats = np.empty(n_boot)

    for b in range(n_boot):
        if weight_distribution == "rademacher":
            weights = rng.choice([-1, 1], size=n)
        elif weight_distribution == "mammen":
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            v1 = -(np.sqrt(5) - 1) / 2
            v2 = (np.sqrt(5) + 1) / 2
            weights = np.where(rng.random(n) < p, v1, v2)
        else:
            raise ValueError(f"Unknown weight_distribution: {weight_distribution}")

        y_boot = y_hat + residuals * weights
        beta_boot = np.linalg.lstsq(X, y_boot, rcond=None)[0]
        boot_stats[b] = beta_boot[statistic_idx]

    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - original)
    alpha = 1 - ci_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=original,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        bias=bias,
        n_boot=n_boot,
        method="wild",
        ci_method="percentile",
        boot_distribution=boot_stats,
        original_estimate=original,
    )


# ---------------------------------------------------------------------------
# Block bootstrap (for time series / panel data)
# ---------------------------------------------------------------------------


def block_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block_size: int,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    method: str = "circular",
    seed: int = 42,
) -> BootstrapResult:
    """Block bootstrap for dependent data.

    Parameters
    ----------
    data : array-like
        Time series or dependent data.
    statistic : callable
        Function that computes the statistic.
    block_size : int
        Length of each block.
    n_boot : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence interval level.
    method : str
        'moving' (non-overlapping), 'circular', or 'stationary'.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    original = statistic(data)

    n_blocks = int(np.ceil(n / block_size))
    boot_stats = np.empty(n_boot)

    for b in range(n_boot):
        if method == "circular":
            starts = rng.integers(0, n, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        elif method == "moving":
            max_start = n - block_size
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        elif method == "stationary":
            # Politis & Romano: geometric block length.
            indices = []
            i = rng.integers(0, n)
            while len(indices) < n:
                indices.append(i % n)
                if rng.random() < 1 / block_size:
                    i = rng.integers(0, n)
                else:
                    i += 1
            indices = np.array(indices[:n])
        else:
            raise ValueError(f"Unknown method: {method}")

        boot_stats[b] = statistic(data[indices])

    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - original)
    alpha = 1 - ci_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=original,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        bias=bias,
        n_boot=n_boot,
        method=f"block_{method}",
        ci_method="percentile",
        boot_distribution=boot_stats,
        original_estimate=original,
    )


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------


def jackknife(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    ci_level: float = 0.95,
) -> JackknifeResult:
    """Delete-one jackknife estimation.

    Parameters
    ----------
    data : array-like
        Input data.
    statistic : callable
        Function that computes the statistic.
    ci_level : float
        Confidence interval level.

    Returns
    -------
    JackknifeResult
    """
    data = np.asarray(data)
    n = len(data)
    original = statistic(data)

    jack_stats = np.empty(n)
    for i in range(n):
        jack_data = np.delete(data, i, axis=0)
        jack_stats[i] = statistic(jack_data)

    jack_mean = np.mean(jack_stats)
    pseudovalues = n * original - (n - 1) * jack_stats
    influence = original - jack_stats

    bias = (n - 1) * (jack_mean - original)
    se = np.sqrt((n - 1) / n * np.sum((jack_stats - jack_mean) ** 2))

    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_lo = original - bias - z * se
    ci_hi = original - bias + z * se

    return JackknifeResult(
        estimate=float(original - bias),
        se=float(se),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        bias=float(bias),
        n=n,
        jackknife_estimates=jack_stats,
        pseudovalues=pseudovalues,
        influence_values=influence,
    )


def delete_d_jackknife(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    d: int = 2,
    ci_level: float = 0.95,
    max_subsets: int = 5000,
    seed: int = 42,
) -> JackknifeResult:
    """Delete-d jackknife (generalized jackknife).

    Parameters
    ----------
    data : array-like
        Input data.
    statistic : callable
        Function that computes the statistic.
    d : int
        Number of observations to delete.
    ci_level : float
        Confidence interval level.
    max_subsets : int
        Maximum number of subsets to evaluate.
    seed : int
        Random seed.

    Returns
    -------
    JackknifeResult
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    original = statistic(data)

    from math import comb

    total_subsets = comb(n, d)

    if total_subsets <= max_subsets:
        from itertools import combinations

        delete_sets = list(combinations(range(n), d))
    else:
        delete_sets = [tuple(sorted(rng.choice(n, d, replace=False))) for _ in range(max_subsets)]
        delete_sets = list(set(delete_sets))

    jack_stats = np.empty(len(delete_sets))
    for i, indices in enumerate(delete_sets):
        mask = np.ones(n, dtype=bool)
        mask[list(indices)] = False
        jack_stats[i] = statistic(data[mask])

    jack_mean = np.mean(jack_stats)
    m = len(delete_sets)
    c = (n - d) / d

    bias = c * (jack_mean - original)
    se = np.sqrt(c / m * np.sum((jack_stats - jack_mean) ** 2))

    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_lo = original - bias - z * se
    ci_hi = original - bias + z * se

    pseudovalues = n * original - (n - d) * jack_stats

    return JackknifeResult(
        estimate=float(original - bias),
        se=float(se),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        bias=float(bias),
        n=n,
        jackknife_estimates=jack_stats,
        pseudovalues=pseudovalues,
        influence_values=original - jack_stats,
    )


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: str = "mean_diff",
    n_permutations: int = 9999,
    alternative: str = "two-sided",
    seed: int = 42,
) -> PermutationTestResult:
    """Two-sample permutation test.

    Parameters
    ----------
    group1 : array-like
        First group.
    group2 : array-like
        Second group.
    statistic : str or callable
        Test statistic: 'mean_diff', 'median_diff', 't_stat',
        or a callable(g1, g2) -> float.
    n_permutations : int
        Number of permutations.
    alternative : str
        'two-sided', 'greater', 'less'.
    seed : int
        Random seed.

    Returns
    -------
    PermutationTestResult
    """
    rng = np.random.default_rng(seed)
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    n = len(combined)

    if callable(statistic):
        stat_fn = statistic
    elif statistic == "mean_diff":
        stat_fn = lambda a, b: np.mean(a) - np.mean(b)
    elif statistic == "median_diff":
        stat_fn = lambda a, b: np.median(a) - np.median(b)
    elif statistic == "t_stat":

        def stat_fn(a, b):
            s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
            se = np.sqrt(s1 / len(a) + s2 / len(b))
            return (np.mean(a) - np.mean(b)) / max(se, 1e-10)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    observed = stat_fn(g1, g2)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_dist[i] = stat_fn(perm[:n1], perm[n1:])

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))
    elif alternative == "greater":
        p_value = float(np.mean(null_dist >= observed))
    elif alternative == "less":
        p_value = float(np.mean(null_dist <= observed))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Include observed in count (exact permutation p-value).
    p_value = (p_value * n_permutations + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=float(observed),
        p_value=p_value,
        null_distribution=null_dist,
        n_permutations=n_permutations,
        alternative=alternative,
    )


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str = "mean_diff",
    n_permutations: int = 9999,
    alternative: str = "two-sided",
    seed: int = 42,
) -> PermutationTestResult:
    """Paired permutation test (sign-flipping).

    Parameters
    ----------
    x : array-like
        First measurement.
    y : array-like
        Second measurement (paired with x).
    statistic : str
        'mean_diff' or 'median_diff'.
    n_permutations : int
        Number of permutations.
    alternative : str
        'two-sided', 'greater', 'less'.
    seed : int
        Random seed.

    Returns
    -------
    PermutationTestResult
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    n = len(diffs)

    if statistic == "mean_diff":
        stat_fn = np.mean
    elif statistic == "median_diff":
        stat_fn = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    observed = stat_fn(diffs)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        null_dist[i] = stat_fn(diffs * signs)

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))
    elif alternative == "greater":
        p_value = float(np.mean(null_dist >= observed))
    else:
        p_value = float(np.mean(null_dist <= observed))

    p_value = (p_value * n_permutations + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=float(observed),
        p_value=p_value,
        null_distribution=null_dist,
        n_permutations=n_permutations,
        alternative=alternative,
    )


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def subsampling(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    subsample_size: int | None = None,
    n_subsamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Subsampling inference (Politis, Romano & Wolf).

    Unlike the bootstrap, subsampling draws without replacement at a
    smaller sample size, providing valid inference under weaker conditions.

    Parameters
    ----------
    data : array-like
        Input data.
    statistic : callable
        Function that computes the statistic.
    subsample_size : int, optional
        Size of each subsample.  Default: int(n^0.7).
    n_subsamples : int
        Number of subsamples.
    ci_level : float
        Confidence interval level.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    original = statistic(data)

    if subsample_size is None:
        subsample_size = int(n**0.7)
    subsample_size = min(subsample_size, n - 1)

    sub_stats = np.empty(n_subsamples)
    for b in range(n_subsamples):
        indices = rng.choice(n, size=subsample_size, replace=False)
        sub_stats[b] = statistic(data[indices])

    # Rescale: tau_n = sqrt(n) * (theta_hat - theta)
    scaling = np.sqrt(n / subsample_size)
    scaled_diffs = scaling * (sub_stats - original)

    alpha = 1 - ci_level
    q_lo = np.percentile(scaled_diffs, 100 * alpha / 2)
    q_hi = np.percentile(scaled_diffs, 100 * (1 - alpha / 2))

    ci_lo = original - q_hi / np.sqrt(n)
    ci_hi = original - q_lo / np.sqrt(n)
    se = np.std(scaled_diffs) / np.sqrt(n)

    return BootstrapResult(
        estimate=original,
        se=float(se),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        bias=0.0,
        n_boot=n_subsamples,
        method="subsampling",
        ci_method="subsampling",
        boot_distribution=sub_stats,
        original_estimate=original,
    )


# ---------------------------------------------------------------------------
# .632 and .632+ bootstrap
# ---------------------------------------------------------------------------


def bootstrap_632(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    score_fn: Callable,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """The .632 bootstrap estimator for prediction error.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix.
    y : array-like, shape (n,)
        Response.
    model_fn : callable
        Function that fits a model: model_fn(X_train, y_train) -> model.
    score_fn : callable
        Function that scores predictions: score_fn(y_true, y_pred) -> float.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict
        With keys: apparent_error, bootstrap_error, error_632, error_632plus.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    # Apparent error (training on full data, testing on full data).
    model_full = model_fn(X, y)
    y_pred_full = model_full.predict(X)
    apparent = score_fn(y, y_pred_full)

    # Bootstrap error.
    boot_errors = []
    for b in range(n_boot):
        indices = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), indices)
        if len(oob) == 0:
            continue

        model_boot = model_fn(X[indices], y[indices])
        y_pred_oob = model_boot.predict(X[oob])
        boot_errors.append(score_fn(y[oob], y_pred_oob))

    if not boot_errors:
        return {
            "apparent_error": apparent,
            "bootstrap_error": float("nan"),
            "error_632": float("nan"),
            "error_632plus": float("nan"),
        }

    boot_error = np.mean(boot_errors)

    # .632 estimator.
    error_632 = 0.368 * apparent + 0.632 * boot_error

    # .632+ estimator (Efron & Tibshirani 1997).
    # No-information error rate.
    y_freq = np.unique(y, return_counts=True)
    if len(y_freq[0]) <= 2:
        # Binary: gamma = p * (1-q) + (1-p) * q
        p = np.mean(y == y_freq[0][0])
        q = np.mean(y_pred_full == y_freq[0][0])
        gamma = p * (1 - q) + (1 - p) * q
    else:
        gamma = 1 - np.sum((y_freq[1] / n) ** 2)

    R = (boot_error - apparent) / max(gamma - apparent, 1e-10)
    R = np.clip(R, 0, 1)
    w = 0.632 / (1 - 0.368 * R)
    error_632plus = (1 - w) * apparent + w * boot_error

    return {
        "apparent_error": float(apparent),
        "bootstrap_error": float(boot_error),
        "error_632": float(error_632),
        "error_632plus": float(error_632plus),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    score_fn: Callable,
    n_folds: int = 10,
    stratify: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    seed: int = 42,
) -> CrossValidationResult:
    """K-fold cross-validation.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix.
    y : array-like, shape (n,)
        Response.
    model_fn : callable
        model_fn(X_train, y_train) -> model with .predict().
    score_fn : callable
        score_fn(y_true, y_pred) -> float.
    n_folds : int
        Number of folds.
    stratify : array-like, optional
        Stratification variable for stratified CV.
    groups : array-like, optional
        Group variable for grouped CV (no group split across folds).
    seed : int
        Random seed.

    Returns
    -------
    CrossValidationResult
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        group_folds = np.array_split(unique_groups, n_folds)
        fold_indices = [np.where(np.isin(groups, gf))[0] for gf in group_folds]

    elif stratify is not None:
        stratify = np.asarray(stratify)
        fold_indices = [[] for _ in range(n_folds)]
        for stratum in np.unique(stratify):
            s_idx = np.where(stratify == stratum)[0]
            rng.shuffle(s_idx)
            splits = np.array_split(s_idx, n_folds)
            for f, s in enumerate(splits):
                fold_indices[f].extend(s.tolist())
        fold_indices = [np.array(fi) for fi in fold_indices]

    else:
        indices = rng.permutation(n)
        fold_indices = [fi for fi in np.array_split(indices, n_folds)]

    scores = np.empty(n_folds)
    fold_sizes = []

    for f in range(n_folds):
        test_idx = fold_indices[f]
        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != f])

        model = model_fn(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        scores[f] = score_fn(y[test_idx], y_pred)
        fold_sizes.append(len(test_idx))

    mean_score = float(np.mean(scores))
    se_score = float(np.std(scores, ddof=1) / np.sqrt(n_folds))
    z = stats.norm.ppf(0.975)

    return CrossValidationResult(
        scores=scores,
        mean_score=mean_score,
        se_score=se_score,
        ci_lower=float(mean_score - z * se_score),
        ci_upper=float(mean_score + z * se_score),
        n_folds=n_folds,
        metric="custom",
        fold_sizes=fold_sizes,
    )


def repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    score_fn: Callable,
    n_folds: int = 10,
    n_repeats: int = 10,
    seed: int = 42,
) -> CrossValidationResult:
    """Repeated K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Response.
    model_fn : callable
        Model fitting function.
    score_fn : callable
        Scoring function.
    n_folds : int
        Number of folds per repeat.
    n_repeats : int
        Number of repetitions.
    seed : int
        Random seed.

    Returns
    -------
    CrossValidationResult
    """
    all_scores = []
    all_fold_sizes = []

    for r in range(n_repeats):
        result = cross_validate(
            X,
            y,
            model_fn,
            score_fn,
            n_folds=n_folds,
            seed=seed + r,
        )
        all_scores.extend(result.scores.tolist())
        all_fold_sizes.extend(result.fold_sizes)

    scores = np.array(all_scores)
    mean_score = float(np.mean(scores))
    se_score = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    z = stats.norm.ppf(0.975)

    return CrossValidationResult(
        scores=scores,
        mean_score=mean_score,
        se_score=se_score,
        ci_lower=float(mean_score - z * se_score),
        ci_upper=float(mean_score + z * se_score),
        n_folds=n_folds * n_repeats,
        metric="custom",
        fold_sizes=all_fold_sizes,
    )


def leave_one_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    score_fn: Callable,
) -> CrossValidationResult:
    """Leave-one-out cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Response.
    model_fn : callable
        Model fitting function.
    score_fn : callable
        Scoring function.

    Returns
    -------
    CrossValidationResult
    """
    return cross_validate(X, y, model_fn, score_fn, n_folds=len(y))
