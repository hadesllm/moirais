# morie.fn — function file (hadesllm/morie)
"""Power to decibels."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def power_to_db(power, **kwargs) -> DescriptiveResult:
    """Convert power to decibels.

    .. math::

        P_{\\text{dB}} = 10 \\cdot \\log_{10}(P)

    Parameters
    ----------
    power : float or array-like
        Power value(s). Must be positive.

    Returns
    -------
    DescriptiveResult
    """
    power = np.asarray(power, dtype=float)
    if np.any(power <= 0):
        raise ValueError("Power must be positive for dB conversion.")
    db = 10.0 * np.log10(power)
    val = float(db) if db.ndim == 0 else float(db[0])
    return DescriptiveResult(
        name="power_to_db",
        value=val,
        extra={"power": power, "db": db},
    )


pwrdb = power_to_db


def cheatsheet() -> str:
    return "power_to_db({}) -> Power to decibels."
