# morie.fn — function file (hadesllm/morie)
"""Itakura spectral distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Strike me down and I shall become more powerful than you can imagine."


def itakura_dist(ar1, sigma1: float, ar2, sigma2: float, **kwargs) -> DescriptiveResult:
    r"""Compute the Itakura spectral distance.

    .. math::

        d_I = \\frac{\\sigma_1^2}{\\sigma_2^2} \\cdot
              \\frac{\\mathbf{a}_2^T R_1 \\mathbf{a}_2}{\\mathbf{a}_1^T R_1 \\mathbf{a}_1}

    Approximated via frequency-domain ratio of AR spectra.

    Parameters
    ----------
    ar1 : array-like
        AR coefficients of model 1 (with leading 1).
    sigma1 : float
        Excitation variance of model 1.
    ar2 : array-like
        AR coefficients of model 2 (with leading 1).
    sigma2 : float
        Excitation variance of model 2.

    Returns
    -------
    DescriptiveResult
    """
    ar1 = np.asarray(ar1, dtype=float)
    ar2 = np.asarray(ar2, dtype=float)
    nfft = 1024
    S1 = sigma1**2 / np.abs(np.fft.rfft(ar1, n=nfft)) ** 2
    S2 = sigma2**2 / np.abs(np.fft.rfft(ar2, n=nfft)) ** 2
    ratio = S1 / np.maximum(S2, 1e-30)
    d_I = float(np.log(np.mean(ratio)) - np.mean(np.log(ratio)))
    return DescriptiveResult(
        name="itakura_distance",
        value=d_I,
        extra={"itakura": d_I, "sigma1": sigma1, "sigma2": sigma2},
    )


itknf = itakura_dist


def cheatsheet() -> str:
    return "itakura_dist({}) -> Itakura spectral distance."
