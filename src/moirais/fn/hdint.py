# moirais.fn — function file (hadesllm/moirais)
"""We are what we repeatedly do. Excellence is a habit. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hdi(samples: np.ndarray, credibility: float = 0.95) -> DescriptiveResult:
    """
    Compute the Highest Density Interval (HDI) from posterior samples.

    The HDI is the narrowest interval containing the specified
    probability mass. For unimodal distributions this is the
    shortest credible interval.

    :param samples: 1-D array of posterior samples.
    :type samples: numpy.ndarray
    :param credibility: Probability mass (e.g. 0.95). Default 0.95.
    :type credibility: float
    :return: DescriptiveResult with HDI lower and upper bounds.
    :rtype: DescriptiveResult
    :raises ValueError: If credibility not in (0, 1).

    References
    ----------
    Kruschke J.K. (2015). *Doing Bayesian Data Analysis*, 2nd ed.
    Academic Press. Chapter 25.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if not 0 < credibility < 1:
        raise ValueError(f"credibility must be in (0, 1), got {credibility}.")
    n = len(samples)
    sorted_s = np.sort(samples)
    n_ci = int(np.ceil(credibility * n))
    widths = sorted_s[n_ci:] - sorted_s[: n - n_ci]
    best = int(np.argmin(widths))
    lo = float(sorted_s[best])
    hi = float(sorted_s[best + n_ci])
    return DescriptiveResult(
        name="hdi",
        value=hi - lo,
        extra={"lower": lo, "upper": hi, "credibility": credibility, "width": hi - lo},
    )


hdint = hdi


def cheatsheet() -> str:
    return "hdi({}) -> Highest Density Interval (Bayesian). 'Let go of your hate.' "
