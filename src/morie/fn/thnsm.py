"""Knowledge itself is power. — Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def snap_estimator(
    data: np.ndarray | list[float],
    *,
    statistic: str = "mean",
    n_snaps: int = 1000,
    seed: int | None = None,
) -> DescriptiveResult:
    """Estimate a statistic and its variance via repeated random half-sampling.

    Each iteration randomly selects exactly half the data (the "snap"),
    computes the statistic, and the distribution of these half-sample
    estimates provides a variance estimate.

    Parameters
    ----------
    data : array-like
        Observed data (1D).
    statistic : str
        "mean", "median", or "std".
    n_snaps : int
        Number of half-sample draws.
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``estimate``, ``half_sample_mean``,
        ``half_sample_se``, ``ci_lower``, ``ci_upper``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 4:
        raise ValueError("data must be 1D with at least 4 elements")

    stat_fns = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda a: float(np.std(a, ddof=1)),
    }
    if statistic not in stat_fns:
        raise ValueError(f"statistic must be one of {list(stat_fns.keys())}")

    fn = stat_fns[statistic]
    full_est = float(fn(x))

    n = len(x)
    half = n // 2
    rng = np.random.default_rng(seed)

    estimates = np.empty(n_snaps)
    for i in range(n_snaps):
        idx = rng.choice(n, half, replace=False)
        estimates[i] = fn(x[idx])

    hs_mean = float(estimates.mean())
    hs_se = float(estimates.std(ddof=1))
    ci_lo = float(np.percentile(estimates, 2.5))
    ci_hi = float(np.percentile(estimates, 97.5))

    return DescriptiveResult(
        name="snap_estimator",
        value={
            "estimate": full_est,
            "half_sample_mean": hs_mean,
            "half_sample_se": hs_se,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        },
        extra={"n": n, "n_snaps": n_snaps, "statistic": statistic},
    )


thnsm = snap_estimator


def cheatsheet() -> str:
    return "snap_estimator({}) -> Random half-sampling estimator. 'Perfectly balanced, as all "
