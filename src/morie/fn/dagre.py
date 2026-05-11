# morie.fn — function file (hadesllm/morie)
"""Diagnostic agreement (Cohen's kappa). 'Judge me by my size, do you?'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def diagnostic_agreement(
    rater1: np.ndarray,
    rater2: np.ndarray,
) -> DescriptiveResult:
    """
    Compute Cohen's kappa and percent agreement between two raters.

    .. math::

        \\kappa = \\frac{p_o - p_e}{1 - p_e}

    where :math:`p_o` is observed agreement and :math:`p_e` is expected
    agreement by chance.

    :param rater1: Array of categorical ratings from rater 1.
    :param rater2: Array of categorical ratings from rater 2.
    :return: DescriptiveResult with kappa as value.
    :raises ValueError: If arrays differ in length or are empty.

    References
    ----------
    Cohen, J. (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement, 20(1), 37--46.
    doi:10.1177/001316446002000104
    """
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    if r1.shape != r2.shape or r1.size == 0:
        raise ValueError("rater1 and rater2 must be non-empty arrays of equal length.")

    categories = np.unique(np.concatenate([r1, r2]))
    n = len(r1)
    p_o = np.sum(r1 == r2) / n

    p_e = 0.0
    for c in categories:
        p_e += (np.sum(r1 == c) / n) * (np.sum(r2 == c) / n)

    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

    se_kappa = np.sqrt(p_o * (1 - p_o) / (n * (1 - p_e) ** 2)) if p_e < 1.0 else 0.0

    return DescriptiveResult(
        name="Cohen's Kappa",
        value=float(np.round(kappa, 4)),
        extra={
            "kappa": float(np.round(kappa, 4)),
            "se": float(np.round(se_kappa, 4)),
            "percent_agreement": float(np.round(p_o, 4)),
            "expected_agreement": float(np.round(p_e, 4)),
            "n": n,
        },
    )


dagre = diagnostic_agreement


def cheatsheet() -> str:
    return "diagnostic_agreement({}) -> Diagnostic agreement (Cohen's kappa). 'Judge me by my size, "
