# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Power spectral density from AR model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def ar_spectrum_fn(
    ar_coeffs: np.ndarray,
    sigma2: float,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Compute power spectral density from AR model parameters.

    .. math::

        S(f) = \\frac{\\sigma^2}{\\left|1 - \\sum_{k=1}^{p} a_k e^{-j 2\\pi f k / f_s}\\right|^2 f_s}

    :param ar_coeffs: AR coefficients [a1, ..., ap].
    :param sigma2: Prediction error variance.
    :param nfft: Number of frequency bins (default 512).
    :param fs: Sampling frequency in Hz (default 1.0).
    :return: DescriptiveResult with frequency vector and PSD.
    """
    from moirais._armodel import ar_spectrum

    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    freqs, psd = ar_spectrum(ar_coeffs, sigma2, fs=fs, n_points=nfft)
    return DescriptiveResult(
        name="ar_spectrum",
        value=None,
        extra={"frequencies": freqs, "psd": psd, "sigma2": sigma2, "fs": fs},
    )


arspd = ar_spectrum_fn


def cheatsheet() -> str:
    return "ar_spectrum_fn({}) -> Power spectral density from AR model."
