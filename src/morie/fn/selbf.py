# morie.fn — function file (hadesllm/morie)
"""Everything flows. — Heraclitus"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def selection_bias_factor(
    p_selected: float,
    outcome_diff: float,
) -> DescriptiveResult:
    """
    Quantify selection bias as a multiplicative factor.

    When selection into a study depends on both treatment and outcome,
    the bias factor approximates the ratio by which the naive estimate
    is inflated.

    .. math::

        BF = \\frac{1}{1 - (1 - P_s) \\cdot \\Delta}

    where :math:`P_s` is the selection probability and :math:`\\Delta` is
    the difference in selection rates by outcome status.

    :param p_selected: Overall probability of selection into the sample, (0, 1].
    :param outcome_diff: Difference in selection probabilities between
        outcome groups (absolute value).
    :return: DescriptiveResult with bias factor.
    :raises ValueError: If parameters out of range.

    References
    ----------
    Haneuse, S., VanderWeele, T. J., & Arterburn, D. (2019). Using the
    E-value to assess the potential effect of unmeasured confounding in
    observational studies. JAMA, 321(6), 602--603. doi:10.1001/jama.2018.21554
    """
    if not 0.0 < p_selected <= 1.0:
        raise ValueError(f"p_selected must be in (0, 1], got {p_selected}.")
    if outcome_diff < 0.0:
        raise ValueError(f"outcome_diff must be >= 0, got {outcome_diff}.")

    denom = 1.0 - (1.0 - p_selected) * outcome_diff
    if denom <= 0:
        bf = float("inf")
    else:
        bf = 1.0 / denom

    return DescriptiveResult(
        name="Selection Bias Factor",
        value=float(np.round(bf, 4)),
        extra={
            "bias_factor": float(np.round(bf, 4)),
            "p_selected": p_selected,
            "outcome_diff": outcome_diff,
        },
    )


selbf = selection_bias_factor


def cheatsheet() -> str:
    return "selection_bias_factor({}) -> Selection bias factor. 'Look well into thyself; there is a source which will always spring up. — Marcus Aurelius' -- "
