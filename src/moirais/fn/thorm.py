"""When I let go of what I am, I become what I might be. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bonferroni_correction(
    p_values: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """Apply Bonferroni correction to a family of p-values.

    Adjusts each p-value by multiplying by *m* (number of tests),
    capping at 1.0.  Reports which hypotheses are rejected at the
    family-wise error rate *alpha*.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from *m* hypothesis tests.
    alpha : float
        Family-wise significance level (default 0.05).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``adjusted`` (array), ``rejected`` (bool array),
        ``n_rejected`` int, ``alpha_adj`` (Bonferroni threshold).
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty 1D array")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("All p-values must be in [0, 1]")

    m = len(p)
    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted < alpha
    alpha_adj = alpha / m

    return DescriptiveResult(
        name="bonferroni_correction",
        value={
            "adjusted": adjusted,
            "rejected": rejected,
            "n_rejected": int(rejected.sum()),
            "alpha_adj": alpha_adj,
        },
        extra={"m": m, "alpha": alpha},
    )


thorm = bonferroni_correction


def cheatsheet() -> str:
    return "bonferroni_correction({}) -> Bonferroni correction for multiple hypothesis testing. 'Brin"
