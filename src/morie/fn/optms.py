# morie.fn -- function file (hadesllm/morie)
"""Knowing others is intelligence; knowing yourself is true wisdom. -- Lao Tzu"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def optimal_transport(
    x: np.ndarray,
    y: np.ndarray,
    *,
    p: int = 1,
) -> DescriptiveResult:
    """Compute the p-Wasserstein distance between two 1-D empirical distributions.

    Uses the closed-form solution for 1-D distributions:
    W_p(F, G) = (integral |F^{-1}(t) - G^{-1}(t)|^p dt)^{1/p}

    Parameters
    ----------
    x, y : array-like
        Samples from two distributions.
    p : int
        Order of the Wasserstein distance (1 or 2).

    Returns
    -------
    DescriptiveResult
        With ``value`` = Wasserstein distance.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Both samples must be non-empty")
    if p not in (1, 2):
        raise ValueError("Only p=1 and p=2 supported")

    if p == 1:
        dist = float(stats.wasserstein_distance(x, y))
    else:
        xs = np.sort(x)
        ys = np.sort(y)
        n = max(len(xs), len(ys))
        q = np.linspace(0, 1, n + 2)[1:-1]
        xq = np.quantile(xs, q)
        yq = np.quantile(ys, q)
        dist = float(np.mean((xq - yq) ** 2) ** 0.5)

    return DescriptiveResult(
        name=f"Wasserstein-{p}",
        value=dist,
        extra={"p": p, "n_x": len(x), "n_y": len(y)},
    )


optms = optimal_transport


def cheatsheet() -> str:
    return "optimal_transport({}) -> Optimal transport / Wasserstein distance. 'Freedom is the ri"
