"""A journey of a thousand miles begins with a single step. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bootstrap_resample(
    x: np.ndarray | list,
    *,
    stat_fn: str = "mean",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Non-parametric bootstrap resampling with percentile CI.

    Draws *n_boot* samples with replacement from *x*, computes *stat_fn*
    on each, and returns the bootstrap distribution summary.

    Parameters
    ----------
    x : array-like
        Original data (1-D).
    stat_fn : str
        Statistic to compute: ``"mean"``, ``"median"``, ``"std"``, ``"var"``.
    n_boot : int
        Number of bootstrap replicates.
    ci : float
        Confidence level for the percentile interval.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the bootstrap estimate (mean of replicates);
        ``extra`` has ``ci_lower``, ``ci_upper``, ``se``, ``bias``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("x must be a 1-D array with at least 2 elements")

    stat_fns = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "var": np.var,
    }
    if stat_fn not in stat_fns:
        raise ValueError(f"stat_fn must be one of {list(stat_fns)}")
    fn = stat_fns[stat_fn]

    rng = np.random.default_rng(seed)
    original_stat = float(fn(arr))
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[b] = fn(sample)

    alpha = 1 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    boot_mean = float(boot_stats.mean())
    boot_se = float(boot_stats.std(ddof=1))

    return DescriptiveResult(
        name=f"Bootstrap ({stat_fn})",
        value=boot_mean,
        extra={
            "original_stat": original_stat,
            "ci_lower": lo,
            "ci_upper": hi,
            "se": boot_se,
            "bias": boot_mean - original_stat,
            "n_boot": n_boot,
            "n": len(arr),
        },
    )


smith = bootstrap_resample


def cheatsheet() -> str:
    return "bootstrap_resample({}) -> Bootstrap resampling with replacement. 'Me, me, me.' -- Agen"
