# morie.fn -- function file (hadesllm/morie)
"""Autocorrelation function of residuals for whiteness testing."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Knowing others is intelligence; knowing yourself is true wisdom. -- Lao Tzu"


def residual_acf_fn(residuals: np.ndarray, nlags: int = 20) -> DescriptiveResult:
    """Compute the autocorrelation function of residuals.

    Used to visually and quantitatively assess whether model residuals
    are white noise (all ACF values within confidence bounds).

    :param residuals: 1-D residual signal.
    :param nlags: Number of lags to compute (default 20).
    :return: DescriptiveResult with ACF values and 95% confidence bounds.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    n = len(residuals)
    r_c = residuals - np.mean(residuals)
    acf_full = np.correlate(r_c, r_c, mode="full")
    acf_full = acf_full[n - 1 :] / acf_full[n - 1]
    acf_vals = acf_full[: nlags + 1]
    ci_bound = 1.96 / np.sqrt(n)
    within_bounds = bool(np.all(np.abs(acf_vals[1:]) < ci_bound))
    return DescriptiveResult(
        name="residual_acf",
        value=float(ci_bound),
        extra={
            "acf": acf_vals,
            "ci_bound": ci_bound,
            "white_noise": within_bounds,
            "nlags": nlags,
            "n": n,
        },
    )


resac = residual_acf_fn


def cheatsheet() -> str:
    return "residual_acf_fn({}) -> Autocorrelation function of residuals for whiteness testing."
