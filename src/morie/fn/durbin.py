# morie.fn — function file (hadesllm/morie)
"""Durbin-Watson autocorrelation test."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def durbin_watson(residuals) -> DescriptiveResult:
    """Durbin-Watson statistic for residual autocorrelation.

    Range [0, 4]: values near 2 indicate no autocorrelation,
    values near 0 indicate positive autocorrelation,
    values near 4 indicate negative autocorrelation.

    Parameters
    ----------
    residuals : array-like
        Regression residuals.

    Returns
    -------
    DescriptiveResult
    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 3:
        raise ValueError("Need at least 3 residuals")
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e**2))
    if dw < 1.5:
        interp = "positive autocorrelation"
    elif dw < 2.5:
        interp = "no autocorrelation"
    else:
        interp = "negative autocorrelation"
    return DescriptiveResult(
        name="Durbin-Watson",
        value=dw,
        extra={"dw": dw, "interpretation": interp},
    )


durbin = durbin_watson


def cheatsheet() -> str:
    return "durbin_watson({}) -> Durbin-Watson autocorrelation test."
