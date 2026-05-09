# moirais.fn — function file (hadesllm/moirais)
"""Group delay of a digital filter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful, the mind of a child is."


def group_delay(b, a, worN: int = 512) -> DescriptiveResult:
    """Compute group delay tau_g(omega) = -d(phi)/d(omega).

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
    from scipy.signal import group_delay as _gd

    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    w, gd = _gd((b, a), w=worN)
    return DescriptiveResult(
        name="group_delay",
        value=float(np.mean(gd)),
        extra={"frequencies": w, "delay": gd},
    )


grpdl = group_delay


def cheatsheet() -> str:
    return "group_delay({}) -> Group delay of a digital filter."
