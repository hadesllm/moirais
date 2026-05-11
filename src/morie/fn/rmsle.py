# morie.fn — function file (hadesllm/morie)
"""RMS log spectral error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def rms_log_error(S1, S2, **kwargs) -> DescriptiveResult:
    """Compute the RMS log spectral error.

    .. math::

        d_{RMS} = \\sqrt{\\frac{1}{K}\\sum_k \\left(\\ln S_1(k) - \\ln S_2(k)\\right)^2}

    Parameters
    ----------
    S1 : array-like
        Power spectrum 1.
    S2 : array-like
        Power spectrum 2.

    Returns
    -------
    DescriptiveResult
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    S1 = np.maximum(S1, 1e-30)
    S2 = np.maximum(S2, 1e-30)
    log_diff = np.log(S1) - np.log(S2)
    rms = float(np.sqrt(np.mean(log_diff**2)))
    return DescriptiveResult(
        name="rms_log_error",
        value=rms,
        extra={"rms_log_error": rms},
    )


rmsle = rms_log_error


def cheatsheet() -> str:
    return "rms_log_error({}) -> RMS log spectral error."
