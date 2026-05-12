# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Region of Practical Equivalence (ROPE) analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bayesian_rope(
    posterior_samples: np.ndarray | list,
    rope_lower: float = -0.1,
    rope_upper: float = 0.1,
) -> DescriptiveResult:
    """
    Evaluate the proportion of the posterior within a ROPE.

    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples.
    rope_lower, rope_upper : float
        ROPE bounds.

    Returns
    -------
    DescriptiveResult
        value = proportion inside ROPE.
        extra has 'pct_below', 'pct_inside', 'pct_above', 'decision'.

    References
    ----------
    Kruschke, J. K. (2018). Rejecting or accepting parameter values
    in Bayesian estimation. *Adv Methods Pract Psychol Sci*, 1(2),
    270-280.
    """
    s = np.asarray(posterior_samples, dtype=float)
    if rope_lower >= rope_upper:
        raise ValueError("rope_lower must be < rope_upper.")
    n = len(s)

    inside = np.sum((s >= rope_lower) & (s <= rope_upper)) / n
    below = np.sum(s < rope_lower) / n
    above = np.sum(s > rope_upper) / n

    if inside > 0.95:
        decision = "accept_null"
    elif inside < 0.05:
        decision = "reject_null"
    else:
        decision = "undecided"

    return DescriptiveResult(
        name="ROPE",
        value=float(inside),
        extra={
            "pct_below": float(below),
            "pct_inside": float(inside),
            "pct_above": float(above),
            "decision": decision,
            "rope": (rope_lower, rope_upper),
            "n": n,
        },
    )


brop = bayesian_rope


def cheatsheet() -> str:
    return "bayesian_rope({}) -> Region of Practical Equivalence (ROPE) analysis."
