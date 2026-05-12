# morie.fn -- function file (hadesllm/morie)
"""Probability manipulation via resampling. 'No more.' -- Scarlet Witch"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def probability_resample(data: np.ndarray | list[float], cdf=None, *, target_quantile: float = 0.5, n_resamples: int = 1000, size: int | None = None, seed: int | None = None) -> DescriptiveResult:
    """Estimate a target quantile via bootstrap resampling with BCa correction.

    Generates *n_resamples* bootstrap samples, computes the target quantile
    in each, and returns the BCa-adjusted confidence interval.

    Parameters
    ----------
    data : array-like
        Observed data (1D).
    target_quantile : float
        Quantile to estimate (0, 1).
    n_resamples : int
        Number of bootstrap resamples.
    size : int or None
        Resample size (default = len(data)).
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``estimate``, ``ci_lower``, ``ci_upper``,
        ``se``, ``bias``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("data must be 1D with at least 2 elements")
    if not 0 < target_quantile < 1:
        raise ValueError("target_quantile must be in (0, 1)")

    n = len(x)
    if size is None:
        size = n
    rng = np.random.default_rng(seed)

    theta_hat = float(np.quantile(x, target_quantile))

    boot_thetas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=size)
        boot_thetas[b] = np.quantile(x[idx], target_quantile)

    se = float(np.std(boot_thetas, ddof=1))
    bias = float(np.mean(boot_thetas) - theta_hat)

    from scipy import stats

    z0 = stats.norm.ppf(np.mean(boot_thetas < theta_hat))
    if not np.isfinite(z0):
        z0 = 0.0

    jack_thetas = np.empty(n)
    for i in range(n):
        jack_thetas[i] = np.quantile(np.delete(x, i), target_quantile)
    jack_mean = jack_thetas.mean()
    num = np.sum((jack_mean - jack_thetas) ** 3)
    den = 6 * (np.sum((jack_mean - jack_thetas) ** 2)) ** 1.5
    a_hat = num / den if abs(den) > 1e-30 else 0.0

    z_alpha = stats.norm.ppf(0.025)
    z_1alpha = stats.norm.ppf(0.975)
    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))
    ci_lo = float(np.quantile(boot_thetas, max(0, min(1, alpha1))))
    ci_hi = float(np.quantile(boot_thetas, max(0, min(1, alpha2))))

    return DescriptiveResult(
        name="probability_resample",
        value={
            "estimate": theta_hat,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "se": se,
            "bias": bias,
        },
        extra={"n": n, "n_resamples": n_resamples, "quantile": target_quantile},
    )


scltw = probability_resample


def cheatsheet() -> str:
    return "probability_resample({}) -> Probability manipulation via resampling. 'No more.' -- Scarl"
