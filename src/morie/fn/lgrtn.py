# morie.fn — function file (hadesllm/morie)
"""Log spectral distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def log_spectral_dist(S1, S2, **kwargs) -> DescriptiveResult:
    """Compute the log spectral distance.

    .. math::

        d_{LS} = \\sqrt{\\frac{1}{K}\\sum_{k}\\left(10\\log_{10}\\frac{S_1(k)}{S_2(k)}\\right)^2}

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
    log_ratio = 10.0 * np.log10(S1 / S2)
    lsd = float(np.sqrt(np.mean(log_ratio**2)))
    return DescriptiveResult(
        name="log_spectral_distance",
        value=lsd,
        extra={"lsd": lsd},
    )


lgrtn = log_spectral_dist


def cheatsheet() -> str:
    return "log_spectral_dist({}) -> Log spectral distance."
