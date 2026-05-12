"""Hope in reality is the worst of all evils because it prolongs the torments of man. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def sample_size_proportion(
    p: float,
    margin: float = 0.05,
    confidence: float = 0.95,
) -> DescriptiveResult:
    r"""
    Compute required sample size for estimating a proportion.

    .. math::

        n = \\frac{z_{\\alpha/2}^{2} \\, p(1-p)}{E^2}

    :param p: Expected proportion, in (0, 1).
    :param margin: Desired margin of error *E*. Default 0.05.
    :param confidence: Confidence level. Default 0.95.
    :return: DescriptiveResult with required *n* (ceiling).
    :raises ValueError: If parameters outside valid ranges.

    References
    ----------
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    if margin <= 0.0:
        raise ValueError(f"margin must be > 0, got {margin}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    z = _st.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    n = (z**2 * p * (1.0 - p)) / (margin**2)
    n_ceil = int(np.ceil(n))

    return DescriptiveResult(
        name="Sample Size for Proportion",
        value=n_ceil,
        extra={
            "n_exact": float(np.round(n, 2)),
            "n_ceiling": n_ceil,
            "z": float(np.round(z, 4)),
            "p": p,
            "margin": margin,
            "confidence": confidence,
        },
    )


srsiz = sample_size_proportion


def cheatsheet() -> str:
    return "sample_size_proportion({}) -> Sample size for a proportion. 'I find your lack of faith dis"
