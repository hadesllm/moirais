# morie.fn — function file (hadesllm/morie)
"""Coiflet wavelet filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Life is really simple, but we insist on making it complicated. — Confucius"


def coiflet_coeffs(order: int = 1) -> DescriptiveResult:
    """Coiflet wavelet filter coefficients.

    Coiflets have vanishing moments for both the scaling and wavelet functions.

    Parameters
    ----------
    order : int
        Coiflet order (1 or 2 supported). Default 1.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``lo_d``, ``hi_d``, ``lo_r``, ``hi_r``.
    """
    _COIF = {
        1: np.array(
            [
                -0.01565573,
                -0.07273262,
                0.38486484,
                0.85257202,
                0.33789767,
                -0.07273262,
            ]
        ),
        2: np.array(
            [
                -0.00072054,
                -0.00182320,
                0.00563443,
                0.02361874,
                -0.05947400,
                -0.07664880,
                0.41700880,
                0.81272320,
                0.38613560,
                -0.06737040,
                -0.04146490,
                0.01643275,
            ]
        ),
    }
    if order not in _COIF:
        raise ValueError(f"Supported coiflet orders: {sorted(_COIF)}; got {order}")
    lo_d = _COIF[order]
    hi_d = np.array([(-1) ** k * lo_d[len(lo_d) - 1 - k] for k in range(len(lo_d))])
    lo_r = lo_d[::-1]
    hi_r = hi_d[::-1]
    return DescriptiveResult(
        name="coiflet_coeffs",
        value=float(order),
        extra={"lo_d": lo_d, "hi_d": hi_d, "lo_r": lo_r, "hi_r": hi_r, "order": order},
    )


coiff = coiflet_coeffs


def cheatsheet() -> str:
    return "coiflet_coeffs({}) -> Coiflet wavelet filter coefficients."
