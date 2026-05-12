# morie.fn -- function file (hadesllm/morie)
"""Linear prediction gain."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficult to see. Always in motion is the future."


def prediction_gain(x, ar_coeffs, **kwargs) -> DescriptiveResult:
    r"""Compute the linear prediction gain.

    .. math::

        G_p = \\frac{P_x}{P_e} = \\frac{\\sigma_x^2}{\\sigma_e^2}

    Parameters
    ----------
    x : array-like
        Input signal.
    ar_coeffs : array-like
        AR coefficients [a1, a2, ...] (without leading 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    ar = np.asarray(ar_coeffs, dtype=float)
    p = len(ar)
    N = len(x)
    if p >= N:
        raise ValueError("Signal length must exceed AR order.")
    pred = np.zeros(N)
    for n in range(p, N):
        pred[n] = np.dot(ar, x[n - p : n][::-1])
    error = x[p:] - pred[p:]
    Px = float(np.var(x))
    Pe = float(np.var(error))
    if Pe <= 0:
        Gp = float("inf")
    else:
        Gp = Px / Pe
    Gp_db = 10.0 * np.log10(max(Gp, 1e-30))
    return DescriptiveResult(
        name="prediction_gain",
        value=float(Gp),
        extra={"Gp": float(Gp), "Gp_db": float(Gp_db), "Px": Px, "Pe": Pe},
    )


predg = prediction_gain


def cheatsheet() -> str:
    return "prediction_gain({}) -> Linear prediction gain."
