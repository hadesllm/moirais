# moirais.fn — function file (hadesllm/moirais)
"""Poles and zeros of a transfer function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Strike me down and I will become more powerful than you can possibly imagine."


def poles_zeros(b, a) -> DescriptiveResult:
    """Compute poles and zeros of a transfer function H(z) = B(z)/A(z).

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.

    Returns
    -------
    DescriptiveResult
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    zeros = np.roots(b)
    poles = np.roots(a)
    gain = b[0] / a[0] if a[0] != 0 else float("inf")
    return DescriptiveResult(
        name="poles_zeros",
        value=float(gain),
        extra={"zeros": zeros, "poles": poles, "gain": gain},
    )


plzro = poles_zeros


def cheatsheet() -> str:
    return "poles_zeros({}) -> Poles and zeros of a transfer function."
