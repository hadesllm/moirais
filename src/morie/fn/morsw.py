# morie.fn — function file (hadesllm/morie)
"""Generalized Morse wavelet."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Confine yourself to the present. — Marcus Aurelius"


def morse_wavelet(
    beta: float = 3.0,
    gamma_param: float = 3.0,
    N: int = 256,
) -> DescriptiveResult:
    r"""Generalized Morse wavelet in the frequency domain.

    .. math::

        \\Psi(\\omega) = U(\\omega) \\, a_{\\beta,\\gamma} \\,
        \\omega^\\beta \\, e^{-\\omega^\\gamma}

    Parameters
    ----------
    beta : float
        Time-decay parameter (default 3.0).
    gamma_param : float
        Frequency-decay parameter (default 3.0).
    N : int
        Number of samples (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wavelet_freq`` (frequency domain),
        ``wavelet_time`` (time domain), ``peak_frequency``.
    """
    omega = np.linspace(0, np.pi, N // 2 + 1)
    peak_freq = (beta / gamma_param) ** (1.0 / gamma_param)
    a = 2 * (np.e * gamma_param / beta) ** (beta / gamma_param)
    psi_freq = np.zeros(N // 2 + 1)
    psi_freq[1:] = a * omega[1:] ** beta * np.exp(-(omega[1:] ** gamma_param))
    psi_full = np.zeros(N)
    psi_full[: N // 2 + 1] = psi_freq
    psi_time = np.real(np.fft.ifft(psi_full))
    return DescriptiveResult(
        name="morse_wavelet",
        value=float(peak_freq),
        extra={
            "wavelet_freq": psi_freq,
            "wavelet_time": psi_time,
            "peak_frequency": float(peak_freq),
            "beta": beta,
            "gamma": gamma_param,
        },
    )


morsw = morse_wavelet


def cheatsheet() -> str:
    return "morse_wavelet({}) -> Generalized Morse wavelet."
