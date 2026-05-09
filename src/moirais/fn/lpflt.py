# moirais.fn — function file (hadesllm/moirais)
"""Butterworth lowpass filter. 'When I let go of what I am, I become what I might be. — Lao Tzu' -- Chirrut Imwe"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ._containers import DescriptiveResult


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> DescriptiveResult:
    """Apply a Butterworth lowpass filter to a signal.

    Uses second-order sections (SOS) for numerical stability and
    zero-phase filtering via forward-backward application.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input time-domain signal.
    cutoff_hz : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (effective order = 2*order due to zero-phase).

    Returns
    -------
    DescriptiveResult
        name='Lowpass Filter', value=cutoff_hz,
        extra has 'filtered' (ndarray), 'cutoff_hz', 'fs',
        'order', 'nyquist', 'attenuation_db_at_cutoff'.

    References
    ----------
    Butterworth, S. (1930). On the Theory of Filter Amplifiers.
    *Wireless Engineer*, 7(6), 536-541.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    nyq = fs / 2.0

    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Require 0 < cutoff_hz ({cutoff_hz}) < nyquist ({nyq})")

    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    filtered = sosfiltfilt(sos, x)

    return DescriptiveResult(
        name="Lowpass Filter",
        value=cutoff_hz,
        extra={
            "filtered": filtered,
            "cutoff_hz": cutoff_hz,
            "fs": fs,
            "order": order,
            "nyquist": nyq,
            "n_samples": len(x),
        },
    )


lpflt = lowpass_filter


def cheatsheet() -> str:
    return "lowpass_filter({}) -> Butterworth lowpass filter. 'When I let go of what I am, I become what I might be. — Lao Tzu' -- Ch"
