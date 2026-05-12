# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""He who would learn to fly one day must first learn to stand and walk. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def bland_altman(
    method1: np.ndarray,
    method2: np.ndarray,
    confidence: float = 0.95,
) -> DescriptiveResult:
    r"""
    Bland-Altman analysis for method comparison agreement.

    Computes the mean difference (bias) and 95% limits of agreement:

    .. math::

        \\text{LoA} = \\bar{d} \\pm z_{\\alpha/2} \\cdot s_d

    :param method1: Measurements from method 1.
    :param method2: Measurements from method 2 (same length).
    :param confidence: Confidence level for limits. Default 0.95.
    :return: DescriptiveResult with bias as value.
    :raises ValueError: If arrays differ in length or are too short.

    References
    ----------
    Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing
    agreement between two methods of clinical measurement. Lancet, 327(8476),
    307--310. doi:10.1016/S0140-6736(86)90837-8
    """
    m1 = np.asarray(method1, dtype=float)
    m2 = np.asarray(method2, dtype=float)
    if m1.shape != m2.shape or m1.ndim != 1 or m1.size < 3:
        raise ValueError("method1 and method2 must be 1-D arrays of equal length >= 3.")

    diff = m1 - m2
    mean_vals = (m1 + m2) / 2.0
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    z = _st.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    loa_lower = bias - z * sd_diff
    loa_upper = bias + z * sd_diff

    return DescriptiveResult(
        name="Bland-Altman",
        value=float(np.round(bias, 4)),
        extra={
            "bias": float(np.round(bias, 4)),
            "sd_diff": float(np.round(sd_diff, 4)),
            "loa_lower": float(np.round(loa_lower, 4)),
            "loa_upper": float(np.round(loa_upper, 4)),
            "differences": diff,
            "means": mean_vals,
            "n": len(m1),
            "confidence": confidence,
        },
    )


bldal = bland_altman


def cheatsheet() -> str:
    return "bland_altman({}) -> Bland-Altman agreement analysis. 'You were the chosen one!' "
