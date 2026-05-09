# moirais.fn — function file (hadesllm/moirais)
"""LPC-derived power spectrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Statistics is the grammar of science. — Karl Pearson"


def lpc_spectrum_fn(
    lpc_coeffs: np.ndarray,
    sigma2: float,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Compute the power spectrum from LPC (all-pole) model parameters.

    .. math::

        S(f) = \\frac{\\sigma^2}{\\left|1 - \\sum_{k=1}^{p} a_k e^{-j2\\pi fk/f_s}\\right|^2}

    :param lpc_coeffs: LPC coefficients [a1, ..., ap].
    :param sigma2: Prediction error variance.
    :param nfft: Number of frequency bins (default 512).
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with frequency vector and PSD.
    """
    from moirais._armodel import ar_spectrum

    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float).ravel()
    freqs, psd = ar_spectrum(lpc_coeffs, sigma2, fs=fs, n_points=nfft)
    return DescriptiveResult(
        name="lpc_spectrum",
        value=None,
        extra={"frequencies": freqs, "psd": psd, "sigma2": sigma2, "fs": fs},
    )


lpcsp = lpc_spectrum_fn


def cheatsheet() -> str:
    return "lpc_spectrum_fn({}) -> LPC-derived power spectrum."
