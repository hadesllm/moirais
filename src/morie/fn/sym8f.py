"""Symlet wavelet filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I will finish what you started."


def symlet_coeffs(order: int = 8) -> DescriptiveResult:
    """Symlet wavelet filter coefficients (near-symmetric Daubechies).

    Parameters
    ----------
    order : int
        Symlet order (2, 4, 6, or 8 supported). Default 8.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``lo_d``, ``hi_d``, ``lo_r``, ``hi_r``.
    """
    _SYM = {
        2: np.array([-0.12940952, 0.22414387, 0.83651630, 0.48296291]),
        4: np.array(
            [
                -0.07576571,
                -0.02963553,
                0.49761867,
                0.80373875,
                0.29785780,
                -0.09921954,
                -0.01260397,
                0.03222310,
            ]
        ),
        6: np.array(
            [
                0.01540410,
                0.00349071,
                -0.11799011,
                -0.04831175,
                0.49105594,
                0.78764689,
                0.33792942,
                -0.07263752,
                -0.02106029,
                0.04472490,
                0.00176771,
                -0.00780070,
            ]
        ),
        8: np.array(
            [
                -0.00338242,
                -0.00054213,
                0.03169509,
                0.00760749,
                -0.14329424,
                -0.06127336,
                0.48135965,
                0.77718575,
                0.36444189,
                -0.05194584,
                -0.02721903,
                0.04913718,
                0.00380875,
                -0.01495226,
                -0.00030292,
                0.00188995,
            ]
        ),
    }
    if order not in _SYM:
        raise ValueError(f"Supported symlet orders: {sorted(_SYM)}; got {order}")
    lo_d = _SYM[order]
    hi_d = np.array([(-1) ** k * lo_d[len(lo_d) - 1 - k] for k in range(len(lo_d))])
    lo_r = lo_d[::-1]
    hi_r = hi_d[::-1]
    return DescriptiveResult(
        name="symlet_coeffs",
        value=float(order),
        extra={"lo_d": lo_d, "hi_d": hi_d, "lo_r": lo_r, "hi_r": hi_r, "order": order},
    )


sym8f = symlet_coeffs


def cheatsheet() -> str:
    return "symlet_coeffs({}) -> Symlet wavelet filter coefficients."
