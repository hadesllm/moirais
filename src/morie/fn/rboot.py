# morie.fn -- function file (hadesllm/morie)
"""Bootstrap CI for reliability."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def bootstrap_reliability(
    data: np.ndarray,
    stat_fn=None,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> ESRes:
    """Bootstrap confidence interval for any reliability statistic.

    Parameters
    ----------
    data : ndarray
        Item response matrix (n x k).
    stat_fn : callable, optional
        Function(data) -> float returning a reliability estimate.
        Default: Cronbach's alpha.
    n_boot : int
        Number of bootstrap resamples (default 1000).
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed (default 42).

    Returns
    -------
    ESRes
        measure="bootstrap_reliability".

    References
    ----------
    Efron, B. & Tibshirani, R. J. (1993). An Introduction to the
    Bootstrap. Chapman & Hall.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    if stat_fn is None:

        def stat_fn(d):
            v_items = np.var(d, axis=0, ddof=1)
            v_total = np.var(d.sum(axis=1), ddof=1)
            if v_total < 1e-15:
                return 0.0
            return (k / (k - 1)) * (1 - v_items.sum() / v_total)

    observed = stat_fn(X)
    rng = np.random.default_rng(seed)
    boot_vals = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_vals[b] = stat_fn(X[idx])

    alpha_half = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_vals, 100 * alpha_half))
    ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha_half)))
    se = float(np.std(boot_vals, ddof=1))

    return ESRes(
        measure="bootstrap_reliability",
        estimate=float(observed),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=n,
        extra={"n_boot": n_boot, "boot_mean": float(np.mean(boot_vals))},
    )


boot_rel = bootstrap_reliability


def cheatsheet() -> str:
    return "bootstrap_reliability({}) -> Bootstrap CI for reliability."
