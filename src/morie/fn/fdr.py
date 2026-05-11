# morie.fn — function file (hadesllm/morie)
"""Benjamini-Hochberg false discovery rate correction."""

import numpy as np

from ._containers import DescriptiveResult


def benjamini_hochberg(
    p_values: np.ndarray | list,
    *,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """Benjamini-Hochberg procedure for controlling the false discovery rate.

    Sort p-values in ascending order. For rank i (1-indexed) among m
    tests, the adjusted p-value is:

        p_adj(i) = min(p(i) * m / i, 1.0)

    enforced to be monotonically non-decreasing from the largest rank
    downward.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from m hypothesis tests.
    alpha : float, default 0.05
        FDR threshold.

    Returns
    -------
    DescriptiveResult
        name="Benjamini-Hochberg", value=number of significant tests,
        extra contains 'adjusted' (array of adjusted p-values in
        original order) and 'significant' (boolean mask).

    Raises
    ------
    ValueError
        If p_values is empty or any value outside [0, 1].

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: A practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
    """
    pv = np.asarray(p_values, dtype=float)
    if len(pv) == 0:
        raise ValueError("p_values must not be empty.")
    if np.any(pv < 0) or np.any(pv > 1):
        raise ValueError("All p-values must be in [0, 1].")

    m = len(pv)
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1, dtype=float)

    sorted_p = pv[order]
    adjusted_sorted = np.minimum(sorted_p * m / ranks, 1.0)

    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    sig = adjusted < alpha

    return DescriptiveResult(
        name="Benjamini-Hochberg",
        value=int(np.sum(sig)),
        extra={
            "adjusted": adjusted.tolist(),
            "significant": sig.tolist(),
            "m": m,
            "alpha": alpha,
        },
    )


fdr = benjamini_hochberg


def cheatsheet() -> str:
    return "benjamini_hochberg({}) -> Benjamini-Hochberg false discovery rate correction."
