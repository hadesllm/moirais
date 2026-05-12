"""Distribution-free tolerance limits (Gibbons Ch 2.11).

Wilks (1941) distribution-free tolerance interval: find sample-size
``n`` such that with confidence ``1-alpha`` at least proportion
``beta`` of the population lies between two order statistics
``X_(r)`` and ``X_(s)``.

Closed-form result for two-sided limits using order statistics
``r=1`` (sample min) and ``s=n`` (sample max):

    P(F(X_(n)) - F(X_(1)) >= beta) = 1 - n*beta^(n-1) + (n-1)*beta^n

The function returns the empirical lower/upper tolerance limits
together with the achieved confidence for the given coverage.

References
----------
Wilks, S. S. (1941). Determination of sample sizes for setting
  tolerance limits. *Ann. Math. Statist.* 12, 91-96.
Gibbons & Chakraborti (6th ed) Ch 2.11.
"""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["tolerance_limits"]


def tolerance_limits(x, coverage: float = 0.90, confidence: float = 0.95):
    """Distribution-free (Wilks) tolerance limits.

    Parameters
    ----------
    x : array-like
        Univariate sample.
    coverage : float in (0, 1)
        Desired population coverage (``beta``).  Default 0.90.
    confidence : float in (0, 1)
        Desired confidence ``1 - alpha`` that the interval covers
        at least ``coverage`` of the population.  Default 0.95.

    Returns
    -------
    RichResult with keys:
        lower, upper        : two-sided tolerance limits (sample min/max)
        coverage_requested  : input ``coverage`` (beta)
        confidence_achieved : achieved confidence for [X_(1), X_(n)]
        n                   : sample size

    References
    ----------
    Gibbons Ch 2.11; Wilks (1941).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "lower": np.nan, "upper": np.nan,
            "coverage_requested": float(coverage),
            "confidence_achieved": np.nan,
            "n": n,
            "method": "Distribution-free tolerance limits (Wilks)",
        })

    beta = float(coverage)
    # Wilks (1941): P(coverage of [X_(1), X_(n)] >= beta)
    #            = 1 - n * beta^(n-1) + (n - 1) * beta^n
    confidence_achieved = 1.0 - n * beta ** (n - 1) + (n - 1) * beta ** n
    confidence_achieved = float(max(0.0, min(1.0, confidence_achieved)))

    lower = float(np.min(x))
    upper = float(np.max(x))
    warnings = []
    if confidence_achieved < confidence:
        warnings.append(
            f"Sample size n={n} too small: achieved confidence "
            f"{confidence_achieved:.4f} < requested {confidence:.4f} "
            f"for coverage {beta:.2f}."
        )
    return RichResult(payload={
        "lower": lower,
        "upper": upper,
        "coverage_requested": beta,
        "confidence_achieved": confidence_achieved,
        "n": n,
        "method": "Distribution-free tolerance limits (Wilks)",
    }, warnings=warnings)


def cheatsheet():
    return "tolim: Distribution-free (Wilks) tolerance limits"


# CANONICAL TEST
# >>> tolerance_limits(list(range(1, 101)), coverage=0.90, confidence=0.95)
# n=100, beta=0.90: achieved confidence = 1 - 100*0.9^99 + 99*0.9^100 ≈ 0.9997
