# morie.fn — function file (hadesllm/morie)
"""Ljung-Box test for white noise."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Train yourself to let go of everything you fear to lose."


def ljung_box_test_fn(residuals: np.ndarray, nlags: int = 20, cdf=None) -> DescriptiveResult:
    """Ljung-Box portmanteau test for residual autocorrelation.

    .. math::

        Q = N(N+2) \\sum_{k=1}^{h} \\frac{\\hat{\\rho}^2(k)}{N-k}

    Under :math:`H_0` (white noise), :math:`Q \\sim \\chi^2(h)`.

    :param residuals: 1-D residual signal.
    :param nlags: Number of lags h (default 20).
    :return: DescriptiveResult with Q statistic and p-value.
    """
    from scipy.stats import chi2

    residuals = np.asarray(residuals, dtype=float).ravel()
    n = len(residuals)
    r_c = residuals - np.mean(residuals)
    acf_full = np.correlate(r_c, r_c, mode="full")
    r0 = acf_full[n - 1]
    if r0 == 0:
        return DescriptiveResult(name="ljung_box", value=0.0, extra={"Q": 0.0, "p_value": 1.0})
    acf_vals = acf_full[n : n + nlags] / r0
    Q = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, nlags + 1)))
    p_value = float(1 - chi2.cdf(Q, df=nlags))
    return DescriptiveResult(
        name="ljung_box",
        value=float(Q),
        extra={"Q": float(Q), "p_value": p_value, "nlags": nlags, "n": n},
    )


ljbx = ljung_box_test_fn


def cheatsheet() -> str:
    return "ljung_box_test_fn({}) -> Ljung-Box test for white noise."
