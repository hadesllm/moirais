"""Circular harmonic analysis. 'You gave me ten rings. I will use them all.' -- Shang-Chi"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ring_harmonics(
    signal: np.ndarray | list[float],
    *,
    n_harmonics: int = 10,
    period: float | None = None,
) -> DescriptiveResult:
    """Decompose a periodic signal into circular harmonics (Fourier on a ring).

    Computes amplitudes and phases for the first *n_harmonics* components
    of a signal assumed to be sampled on a circle.

    Parameters
    ----------
    signal : array-like
        Signal values sampled uniformly on [0, 2pi).
    n_harmonics : int
        Number of harmonic components to extract.
    period : float or None
        Signal period. If None, assumes one full cycle.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``amplitudes``, ``phases``, ``dc_offset``,
        ``power_spectrum``, ``dominant_harmonic``.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 4:
        raise ValueError("signal must be 1D with at least 4 points")

    N = len(x)
    if period is None:
        period = float(N)

    n_harmonics = min(n_harmonics, N // 2)
    fft_vals = np.fft.rfft(x)

    dc = float(np.abs(fft_vals[0])) / N
    amplitudes = 2 * np.abs(fft_vals[1 : n_harmonics + 1]) / N
    phases = np.angle(fft_vals[1 : n_harmonics + 1])
    power = amplitudes**2

    dominant = int(np.argmax(amplitudes)) + 1

    return DescriptiveResult(
        name="ring_harmonics",
        value={
            "amplitudes": amplitudes,
            "phases": phases,
            "dc_offset": dc,
            "power_spectrum": power,
            "dominant_harmonic": dominant,
        },
        extra={"n_harmonics": n_harmonics, "N": N, "period": period},
    )


shang = ring_harmonics


def cheatsheet() -> str:
    return "ring_harmonics({}) -> Circular harmonic analysis. 'You gave me ten rings. I will u"
