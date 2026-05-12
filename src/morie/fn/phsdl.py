# morie.fn -- function file (hadesllm/morie)
"""Phase delay of a digital filter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Much to learn, you still have."


def phase_delay(b, a, worN: int = 512) -> DescriptiveResult:
    """Compute phase delay tau_p(omega) = -phi(omega)/omega.

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.
    worN : int
        Number of frequency points. Default 512.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import freqz

    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    w, h = freqz(b, a, worN=worN)
    phase = -np.unwrap(np.angle(h))
    with np.errstate(divide="ignore", invalid="ignore"):
        pd = np.where(w > 0, phase / w, 0.0)
    return DescriptiveResult(
        name="phase_delay",
        value=float(np.mean(pd[w > 0])) if np.any(w > 0) else 0.0,
        extra={"frequencies": w, "delay": pd},
    )


phsdl = phase_delay


def cheatsheet() -> str:
    return "phase_delay({}) -> Phase delay of a digital filter."
