# morie.fn -- function file (hadesllm/morie)
"""Gabor logon (Gaussian-windowed sinusoid, TF atom)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We had each other. That's how we won."


def gabor_logon(
    t: np.ndarray,
    f0: float,
    sigma: float,
) -> DescriptiveResult:
    r"""Gabor logon: Gaussian-windowed complex sinusoid.

    .. math::

        g(t) = \\frac{1}{\\sigma\\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{t^2}{2\\sigma^2}\\right)
               \\exp(j 2\\pi f_0 t)

    Achieves the minimum TF uncertainty (Heisenberg limit).

    Parameters
    ----------
    t : array-like
        Time vector.
    f0 : float
        Center frequency in Hz.
    sigma : float
        Gaussian width (time spread) in seconds.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``signal``, ``envelope``, ``real``, ``imag``.
    """
    t = np.asarray(t, dtype=float).ravel()
    envelope = np.exp(-(t**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    signal = envelope * np.exp(1j * 2 * np.pi * f0 * t)
    return DescriptiveResult(
        name="gabor_logon",
        value=float(f0),
        extra={
            "signal": signal,
            "envelope": envelope,
            "real": np.real(signal),
            "imag": np.imag(signal),
        },
    )


gborl = gabor_logon


def cheatsheet() -> str:
    return "gabor_logon({}) -> Gabor logon (Gaussian-windowed sinusoid, TF atom)."
