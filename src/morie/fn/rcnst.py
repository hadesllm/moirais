# morie.fn — function file (hadesllm/morie)
"""Whittaker-Shannon sinc interpolation."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "All models are wrong, but some are useful. — George E. P. Box"


def sinc_reconstruct(samples, fs: float, t_new) -> SignalResult:
    """Reconstruct a continuous-time signal using sinc interpolation.

    .. math::

        x(t) = \\sum_{n} x(n) \\, \\text{sinc}\\left(\\frac{t - n/f_s}{1/f_s}\\right)

    Parameters
    ----------
    samples : array-like
        Discrete samples x(n).
    fs : float
        Sampling frequency (Hz).
    t_new : array-like
        Time points at which to reconstruct.

    Returns
    -------
    SignalResult
    """
    samples = np.asarray(samples, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    T = 1.0 / fs
    n_samples = len(samples)
    t_n = np.arange(n_samples) * T
    sinc_matrix = np.sinc((t_new[:, None] - t_n[None, :]) / T)
    y = sinc_matrix @ samples
    return SignalResult(
        name="sinc_reconstruct",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"t_new": t_new, "n_original": n_samples},
    )


rcnst = sinc_reconstruct


def cheatsheet() -> str:
    return "sinc_reconstruct({}) -> Whittaker-Shannon sinc interpolation."
