"""Wiener-Hopf optimal filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wiener_hopf_solve(Rxx, rxd) -> DescriptiveResult:
    """Solve the Wiener-Hopf equation for optimal filter coefficients.

    Parameters
    ----------
    Rxx : array-like
        Autocorrelation matrix.
    rxd : array-like
        Cross-correlation vector.

    Returns
    -------
    DescriptiveResult
    """
    from moirais._filters import wiener_hopf_solve as _wh

    Rxx = np.asarray(Rxx, dtype=float)
    rxd = np.asarray(rxd, dtype=float)
    coeffs = _wh(Rxx, rxd)
    return DescriptiveResult(
        name="wiener_hopf",
        value=len(coeffs),
        extra={"coefficients": coeffs},
    )


wnhpf = wiener_hopf_solve


def cheatsheet() -> str:
    return "wiener_hopf_solve({}) -> Wiener-Hopf optimal filter coefficients."
