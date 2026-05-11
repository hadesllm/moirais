"""Cross-wavelet transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def _morlet(t, omega0=6.0):
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2)


def wavelet_cross_spectrum(x, y, scales=None) -> DescriptiveResult:
    """Cross-wavelet transform between two signals.

    Parameters
    ----------
    x : array-like
        First signal.
    y : array-like
        Second signal.
    scales : array-like or None
        CWT scales. Auto if None.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N = min(len(x), len(y))
    x, y = x[:N], y[:N]
    if scales is None:
        scales = np.arange(1, min(N // 2, 64) + 1, dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)

    cwt_x = np.zeros((len(scales), N), dtype=complex)
    cwt_y = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        half = int(4 * s)
        t_w = np.arange(-half, half + 1) / s
        psi = _morlet(t_w) / np.sqrt(s)
        cx = np.convolve(x, np.conj(psi), mode="same")
        cy = np.convolve(y, np.conj(psi), mode="same")
        cwt_x[i, :] = cx[:N]
        cwt_y[i, :] = cy[:N]

    cross = cwt_x * np.conj(cwt_y)
    power = np.abs(cross)
    phase = np.angle(cross)

    return DescriptiveResult(
        name="wavelet_cross_spectrum",
        value=float(np.max(power)),
        extra={
            "cross_power": power,
            "cross_phase": phase,
            "scales": scales,
            "cwt_x": np.abs(cwt_x),
            "cwt_y": np.abs(cwt_y),
        },
    )


wvcrs = wavelet_cross_spectrum


def cheatsheet() -> str:
    return "_morlet({}) -> Cross-wavelet transform."
