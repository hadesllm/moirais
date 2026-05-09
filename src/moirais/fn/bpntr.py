# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""When I let go of what I am, I become what I might be. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def vibranium_damping(
    signal: np.ndarray,
    *,
    fs: float = 1.0,
    damping_ratio: float = 0.1,
    natural_freq: float = 1.0,
) -> DescriptiveResult:
    """Model signal attenuation through a second-order damped system.

    Applies the transfer function of a single-degree-of-freedom oscillator
    :math:`H(s) = \\omega_n^2 / (s^2 + 2\\zeta\\omega_n s + \\omega_n^2)`
    in the frequency domain.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency in Hz.
    damping_ratio : float
        Damping ratio :math:`\\zeta` (0 = undamped, 1 = critically damped).
    natural_freq : float
        Natural frequency :math:`\\omega_n` in Hz.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``output`` (attenuated signal), ``gain_db``
        (frequency response in dB), ``freqs`` (frequency axis).
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if damping_ratio < 0:
        raise ValueError("damping_ratio must be >= 0")

    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    omega = 2 * np.pi * freqs
    omega_n = 2 * np.pi * natural_freq
    zeta = damping_ratio

    denom = (omega_n**2 - omega**2) + 1j * 2 * zeta * omega_n * omega
    denom[np.abs(denom) < 1e-30] = 1e-30
    H = omega_n**2 / denom

    Y = X * H
    output = np.fft.irfft(Y, n=N)

    gain = np.abs(H)
    gain[gain < 1e-30] = 1e-30
    gain_db = 20 * np.log10(gain)

    return DescriptiveResult(
        name="vibranium_damping",
        value={
            "output": output,
            "gain_db": gain_db,
            "freqs": freqs,
        },
        extra={"fs": fs, "damping_ratio": damping_ratio, "natural_freq": natural_freq, "n": N},
    )


bpntr = vibranium_damping


def cheatsheet() -> str:
    return "When I let go of what I am, I become what I might be. — Lao Tzu"
