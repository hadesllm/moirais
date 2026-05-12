# morie.fn -- function file (hadesllm/morie)
"""Bootstrap convergence diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["sbcnv"]


def sbcnv(
    x: np.ndarray,
    statistic: callable | None = None,
    *,
    n_boot_sizes: np.ndarray | None = None,
    n_reps: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Diagnose bootstrap convergence by comparing bootstrap distributions
    at increasing sample sizes.

    Computes bootstrap variance at each size and checks for stabilization
    via the coefficient of variation of successive variance estimates.

    :param x: 1-D array of observations.
    :param statistic: Function(sample) -> scalar. Default: mean.
    :param n_boot_sizes: Array of bootstrap sizes to test. Default geometric sequence.
    :param n_reps: Replications per size. Default 200.
    :param seed: Random seed.
    :return: Dict with ``sizes``, ``variances``, ``cv_ratio``, ``converged``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 19-20. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    rng = np.random.default_rng(seed)

    if statistic is None:
        def statistic(s):
            return float(np.mean(s))

    if n_boot_sizes is None:
        n_boot_sizes = np.unique(np.geomspace(50, min(5000, max(n * 10, 100)), 8).astype(int))

    variances = np.zeros(len(n_boot_sizes))
    for k, B in enumerate(n_boot_sizes):
        boot_stats = np.zeros(B)
        for b in range(B):
            idx = rng.choice(n, size=n, replace=True)
            boot_stats[b] = statistic(x[idx])
        variances[k] = float(np.var(boot_stats, ddof=1))

    if len(variances) >= 3:
        diffs = np.abs(np.diff(variances))
        cv_ratio = float(np.std(diffs) / (np.mean(diffs) + 1e-12))
    else:
        cv_ratio = float("inf")

    converged = cv_ratio < 0.1

    return {
        "sizes": n_boot_sizes,
        "variances": variances,
        "cv_ratio": cv_ratio,
        "converged": converged,
        "n": n,
    }


def cheatsheet() -> str:
    return "sbcnv({x}) -> Bootstrap convergence diagnostic."
