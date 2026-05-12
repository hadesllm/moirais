"""SNR improvement via synchronized averaging."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def snr_improvement(M, **kwargs) -> DescriptiveResult:
    r"""Compute the SNR improvement factor for synchronized averaging.

    .. math::

        \\text{SNR}_{\\text{improvement}} = \\sqrt{M}

    where *M* is the number of synchronized sweeps averaged.

    Parameters
    ----------
    M : int or float
        Number of sweeps / epochs averaged.

    Returns
    -------
    DescriptiveResult
    """
    M = float(M)
    if M <= 0:
        raise ValueError("M must be positive")
    improvement = float(np.sqrt(M))
    return DescriptiveResult(
        name="snr_improvement",
        value=improvement,
        extra={"M": M, "improvement_factor": improvement, "improvement_db": float(10.0 * np.log10(M))},
    )


ssnri = snr_improvement


def cheatsheet() -> str:
    return "snr_improvement({}) -> SNR improvement via synchronized averaging."
