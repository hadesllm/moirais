"""Signal-to-noise ratio."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def snr_compute(signal, noise, **kwargs) -> DescriptiveResult:
    """Compute the signal-to-noise ratio in decibels.

    .. math::

        \\text{SNR} = 10 \\cdot \\log_{10}\\left(\\frac{P_s}{P_n}\\right) \\; \\text{dB}

    Parameters
    ----------
    signal : array-like
        Clean signal.
    noise : array-like
        Noise component.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    ps = float(np.mean(signal**2))
    pn = float(np.mean(noise**2))
    if pn == 0.0:
        snr_db = float("inf")
    else:
        snr_db = float(10.0 * np.log10(ps / pn))
    return DescriptiveResult(
        name="snr_compute",
        value=snr_db,
        extra={"snr_db": snr_db, "power_signal": ps, "power_noise": pn},
    )


ssnr = snr_compute


def cheatsheet() -> str:
    return "snr_compute({}) -> Signal-to-noise ratio."
