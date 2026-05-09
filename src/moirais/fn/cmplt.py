# moirais.fn — function file (hadesllm/moirais)
"""Completeness of ascertainment via capture-recapture."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def capture_recapture(
    n1: int,
    n2: int,
    m: int,
    confidence: float = 0.95,
) -> ESRes:
    """Estimate population size using two-source capture-recapture.

    Uses the Chapman estimator (bias-corrected Lincoln-Petersen).

    .. math::

        \\hat{N} = \\frac{(n_1 + 1)(n_2 + 1)}{m + 1} - 1

    Parameters
    ----------
    n1 : int
        Cases captured by source 1.
    n2 : int
        Cases captured by source 2.
    m : int
        Cases captured by both sources.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Chapman, D. G. (1951). Some properties of the hypergeometric
    distribution with applications to zoological sample censuses.
    University of California Publications in Statistics, 1, 131-160.

    Hook, E. B. & Regal, R. R. (1995). Capture-recapture methods in
    epidemiology. Epidemiologic Reviews, 17(2), 243-264.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    if m < 0 or m > min(n1, n2):
        raise ValueError("m must be in [0, min(n1, n2)]")
    if m == 0:
        raise ValueError("m must be > 0 for estimation")

    N_hat = ((n1 + 1) * (n2 + 1)) / (m + 1) - 1
    var_N = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)) / ((m + 1) ** 2 * (m + 2))
    se = np.sqrt(var_N)

    import scipy.stats as st
    z = st.norm.ppf((1 + confidence) / 2)

    observed = n1 + n2 - m
    completeness = observed / N_hat if N_hat > 0 else 1.0

    return ESRes(
        measure="capture_recapture",
        estimate=float(N_hat),
        se=float(se),
        ci_lower=float(N_hat - z * se),
        ci_upper=float(N_hat + z * se),
        n=observed,
        extra={
            "completeness": float(completeness),
            "n1": n1,
            "n2": n2,
            "m": m,
            "observed": observed,
        },
    )


cmplt = capture_recapture


def cheatsheet() -> str:
    return "capture_recapture({}) -> Completeness of ascertainment (two-source capture-recapture)."
