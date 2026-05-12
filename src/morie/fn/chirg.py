# morie.fn -- function file (hadesllm/morie)
"""Generate linear/quadratic chirp signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is where the fun begins."


def chirp_generate(
    f0: float,
    f1: float,
    T: float,
    fs: float,
    method: str = "linear",
) -> DescriptiveResult:
    """Generate a chirp (frequency sweep) signal.

    Parameters
    ----------
    f0 : float
        Start frequency in Hz.
    f1 : float
        End frequency in Hz.
    T : float
        Duration in seconds.
    fs : float
        Sampling frequency in Hz.
    method : str
        'linear' or 'quadratic' (default 'linear').

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``signal``, ``time``, ``f0``, ``f1``.
    """
    t = np.arange(0, T, 1.0 / fs)
    if method == "linear":
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2)
    elif method == "quadratic":
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (3 * T**2) * t**3)
    else:
        raise ValueError(f"method must be 'linear' or 'quadratic', got '{method}'")
    signal = np.cos(phase)
    return DescriptiveResult(
        name="chirp_generate",
        value=float(len(t)),
        extra={"signal": signal, "time": t, "f0": f0, "f1": f1},
    )


chirg = chirp_generate


def cheatsheet() -> str:
    return "chirp_generate({}) -> Generate linear/quadratic chirp signal."
