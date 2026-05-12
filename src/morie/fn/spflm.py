"""Spectral flatness (Wiener entropy)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies... with thunderous applause."


def spectral_flatness(x, **kwargs) -> DescriptiveResult:
    r"""Compute spectral flatness (Wiener entropy).

    .. math::

        SF = \\frac{\\exp\\left(\\frac{1}{N}\\sum \\ln S(k)\\right)}
             {\\frac{1}{N}\\sum S(k)}

    Values near 1 indicate white noise; near 0 indicate tonal signal.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2
    psd = psd[psd > 0]
    if len(psd) == 0:
        return DescriptiveResult(name="spectral_flatness", value=0.0, extra={"flatness": 0.0})
    geo_mean = float(np.exp(np.mean(np.log(psd))))
    arith_mean = float(np.mean(psd))
    flatness = geo_mean / arith_mean if arith_mean > 0 else 0.0
    return DescriptiveResult(
        name="spectral_flatness",
        value=flatness,
        extra={"flatness": flatness, "geo_mean": geo_mean, "arith_mean": arith_mean},
    )


spflm = spectral_flatness


def cheatsheet() -> str:
    return "spectral_flatness({}) -> Spectral flatness (Wiener entropy)."
