# moirais.fn — function file (hadesllm/moirais)
"""Dicrotic notch detection in pulse waveforms."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dicrotic_notch_detect(pulse, fs: float = 125.0) -> DescriptiveResult:
    """Detect dicrotic notches in a pulse waveform.

    Parameters
    ----------
    pulse : array-like
        Pulse waveform signal.
    fs : float
        Sampling frequency in Hz. Default 125.

    Returns
    -------
    DescriptiveResult
    """
    from moirais._detection import dicrotic_notch_detect as _dn

    pulse = np.asarray(pulse, dtype=float)
    notches = _dn(pulse, fs=fs)
    return DescriptiveResult(
        name="dicrotic_notch",
        value=len(notches),
        extra={"notch_indices": notches},
    )


dcrtc = dicrotic_notch_detect


def cheatsheet() -> str:
    return "dicrotic_notch_detect({}) -> Dicrotic notch detection in pulse waveforms."
