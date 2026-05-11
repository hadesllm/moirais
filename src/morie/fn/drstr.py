# morie.fn — function file (hadesllm/morie)
"""Parallel universes bootstrap. 'I went forward in time to view alternate futures.' -- Dr. Strange"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def multiverse_bootstrap(
    data: np.ndarray | list[float],
    *,
    n_universes: int = 1000,
    statistic: str = "mean",
    confidence: float = 0.95,
    seed: int | None = None,
) -> DescriptiveResult:
    """Run parallel-universes bootstrap: generate many independent bootstrap
    replicates in parallel (vectorised), returning the full distribution of
    the statistic.

    Parameters
    ----------
    data : array-like
        Observed data (1D).
    n_universes : int
        Number of bootstrap replicates ("parallel universes").
    statistic : str
        "mean", "median", "std", or "trimmed_mean".
    confidence : float
        Confidence level for percentile CI.
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``estimate``, ``distribution`` (array of
        replicate values), ``ci_lower``, ``ci_upper``, ``se``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("data must be 1D with at least 2 elements")

    fns = {
        "mean": lambda a: np.mean(a, axis=1),
        "median": lambda a: np.median(a, axis=1),
        "std": lambda a: np.std(a, axis=1, ddof=1),
        "trimmed_mean": lambda a: np.mean(np.sort(a, axis=1)[:, int(0.1 * a.shape[1]) : int(0.9 * a.shape[1])], axis=1),
    }
    if statistic not in fns:
        raise ValueError(f"Unknown statistic: {statistic}")

    rng = np.random.default_rng(seed)
    n = len(x)
    indices = rng.integers(0, n, size=(n_universes, n))
    boot_samples = x[indices]
    dist = fns[statistic](boot_samples)

    est = float(fns[statistic](x.reshape(1, -1))[0])
    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(dist, 100 * alpha))
    ci_hi = float(np.percentile(dist, 100 * (1 - alpha)))
    se = float(np.std(dist, ddof=1))

    return DescriptiveResult(
        name="multiverse_bootstrap",
        value={
            "estimate": est,
            "distribution": dist,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "se": se,
        },
        extra={"n": n, "n_universes": n_universes, "statistic": statistic, "confidence": confidence},
    )


drstr = multiverse_bootstrap


def cheatsheet() -> str:
    return "multiverse_bootstrap({}) -> Parallel universes bootstrap. 'I went forward in time to vie"
