# moirais.fn — function file (hadesllm/moirais)
"""Region of Practical Equivalence (ROPE) analysis."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def rope_test(
    samples: Union[list, np.ndarray],
    *,
    rope_low: float = -0.1,
    rope_high: float = 0.1,
) -> dict[str, Any]:
    """
    Region of Practical Equivalence (ROPE) decision rule.

    Computes what proportion of the posterior distribution falls within the
    ROPE [rope_low, rope_high].

    Decision rule (Kruschke, 2018):
      * If 95% HDI entirely inside ROPE: **accept** null (practically equivalent)
      * If 95% HDI entirely outside ROPE: **reject** null (practically different)
      * Otherwise: **undecided**

    :param samples: 1-D array of posterior samples.
    :param rope_low: Lower bound of ROPE.
    :param rope_high: Upper bound of ROPE.
    :return: Dictionary with proportion_in_rope, proportion_below,
        proportion_above, decision.
    :raises ValueError: If rope_low >= rope_high or samples is empty.

    References
    ----------
    Kruschke, J. K. (2018). Rejecting or accepting parameter values in
    Bayesian estimation. *Advances in Methods and Practices in
    Psychological Science*, 1(2), 270--280.
    """
    if rope_low >= rope_high:
        raise ValueError("rope_low must be less than rope_high.")
    arr = np.asarray(samples, dtype=float)
    if len(arr) == 0:
        raise ValueError("samples must not be empty.")

    in_rope = np.mean((arr >= rope_low) & (arr <= rope_high))
    below = np.mean(arr < rope_low)
    above = np.mean(arr > rope_high)

    # HDI-based decision
    from moirais.fn.hdi import highest_density_interval

    hdi_lo, hdi_hi = highest_density_interval(arr, mass=0.95)

    if hdi_lo >= rope_low and hdi_hi <= rope_high:
        decision = "accept"
    elif hdi_hi < rope_low or hdi_lo > rope_high:
        decision = "reject"
    else:
        decision = "undecided"

    return {
        "proportion_in_rope": float(in_rope),
        "proportion_below": float(below),
        "proportion_above": float(above),
        "hdi_low": hdi_lo,
        "hdi_high": hdi_hi,
        "decision": decision,
    }


rope = rope_test


def cheatsheet() -> str:
    return "rope_test({}) -> Region of Practical Equivalence (ROPE) analysis."
