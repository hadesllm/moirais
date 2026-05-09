# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""James-Stein shrinkage estimator. 'It's a small world after all.' -- The Atom"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def james_stein(
    x: np.ndarray | list[float],
    *,
    target: float | None = None,
) -> DescriptiveResult:
    """James-Stein shrinkage estimator for a multivariate normal mean.

    Shrinks the observed means toward a common target, reducing total MSE
    when estimating >= 3 means simultaneously (Stein's paradox).

    Parameters
    ----------
    x : array
        Observed sample means (length >= 3).
    target : float or None
        Shrinkage target.  If None, uses the grand mean.

    Returns
    -------
    DescriptiveResult
        ``value`` = shrinkage factor (0 = full shrinkage, 1 = no shrinkage).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = len(x)
    if p < 3:
        raise ValueError("James-Stein requires >= 3 means (Stein's paradox)")
    if target is None:
        target = float(np.mean(x))
    diff = x - target
    ss = float(np.sum(diff**2))
    if ss < 1e-30:
        shrinkage = 0.0
    else:
        shrinkage = max(0.0, 1.0 - (p - 2) / ss)
    js_estimate = target + shrinkage * diff
    return DescriptiveResult(
        name="James-Stein shrinkage estimator",
        value=float(shrinkage),
        extra={
            "p": p,
            "target": target,
            "shrinkage_factor": shrinkage,
            "js_estimates": js_estimate.tolist(),
            "original_means": x.tolist(),
            "mse_reduction_bound": round(1 - shrinkage**2, 4) if shrinkage < 1 else 0.0,
        },
    )


atoms = james_stein


def cheatsheet() -> str:
    return "james_stein({}) -> James-Stein shrinkage estimator. 'It's a small world after a"
