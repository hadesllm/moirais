# morie.fn — function file (hadesllm/morie)
"""Complex Morlet wavelet."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Patience is bitter, but its fruit is sweet. — Aristotle"


def cmor_wavelet(
    fb: float = 1.5,
    fc: float = 1.0,
    N: int = 256,
) -> DescriptiveResult:
    """Complex Morlet wavelet.

    .. math::

        \\psi(t) = \\frac{1}{\\sqrt{\\pi f_b}}
                   \\exp\\!\\left(-\\frac{t^2}{f_b}\\right)
                   \\exp(j 2\\pi f_c t)

    Parameters
    ----------
    fb : float
        Bandwidth parameter (default 1.5).
    fc : float
        Center frequency (default 1.0).
    N : int
        Number of samples (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wavelet``, ``time``, ``real``, ``imag``.
    """
    t = np.linspace(-4, 4, N)
    envelope = np.exp(-(t**2) / fb) / np.sqrt(np.pi * fb)
    psi = envelope * np.exp(1j * 2 * np.pi * fc * t)
    return DescriptiveResult(
        name="cmor_wavelet",
        value=float(fc),
        extra={
            "wavelet": psi,
            "time": t,
            "real": np.real(psi),
            "imag": np.imag(psi),
            "fb": fb,
            "fc": fc,
        },
    )


cmorw = cmor_wavelet


def cheatsheet() -> str:
    return "cmor_wavelet({}) -> Complex Morlet wavelet."
