# moirais.fn — function file (hadesllm/moirais)
"""Unbiased sample variance — denominator (n−1)."""

from typing import Sequence, Union

import numpy as np


def s2var(x: Union[Sequence[float], np.ndarray]) -> float:
    """Unbiased sample variance with Bessel's correction.

    s² = Σᵢ (xᵢ − x̄)² / (n − 1)

    The (n−1) divisor (rather than n) makes the estimator unbiased for the
    population variance. This is the canonical "sample variance" used
    throughout introductory crim-stats. For the population variance (n
    divisor) use `popvar`.

    :param x: numeric sample of size ≥ 2.
    :return: sample variance.

    References
    ----------
    Wooditch et al. (2021). A Beginner's Guide … ch.5 (Variance), eq. 5.x.
    """
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        raise ValueError("need at least 2 observations for sample variance.")
    return float(a.var(ddof=1))
