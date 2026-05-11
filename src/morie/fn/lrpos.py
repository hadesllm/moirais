# morie.fn — function file (hadesllm/morie)
"""Likelihood ratios (LR+ / LR-). 'So this is how liberty dies.' -- Padme Amidala"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def likelihood_ratio(
    sensitivity: float,
    specificity: float,
) -> DescriptiveResult:
    """
    Compute positive (LR+) and negative (LR-) likelihood ratios.

    .. math::

        LR^{+} = \\frac{Se}{1 - Sp}, \\qquad
        LR^{-} = \\frac{1 - Se}{Sp}

    :param sensitivity: Test sensitivity in [0, 1].
    :param specificity: Test specificity in [0, 1].
    :return: DescriptiveResult with LR+ as value, LR- in extra.
    :raises ValueError: If parameters are outside [0, 1] or denominators are zero.

    References
    ----------
    Deeks, J. J., & Altman, D. G. (2004). Diagnostic tests 4: likelihood
    ratios. BMJ, 329(7458), 168--169. doi:10.1136/bmj.329.7458.168
    """
    for name, val in [("sensitivity", sensitivity), ("specificity", specificity)]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}.")
    if specificity == 1.0:
        raise ValueError("specificity=1.0 makes LR+ undefined (division by zero).")
    if specificity == 0.0:
        raise ValueError("specificity=0.0 makes LR- undefined (division by zero).")

    lr_pos = sensitivity / (1.0 - specificity)
    lr_neg = (1.0 - sensitivity) / specificity

    return DescriptiveResult(
        name="Likelihood Ratios",
        value=float(np.round(lr_pos, 4)),
        extra={
            "lr_positive": float(np.round(lr_pos, 4)),
            "lr_negative": float(np.round(lr_neg, 4)),
            "sensitivity": sensitivity,
            "specificity": specificity,
        },
    )


lrpos = likelihood_ratio


def cheatsheet() -> str:
    return "likelihood_ratio({}) -> Likelihood ratios (LR+ / LR-). 'So this is how liberty dies."
