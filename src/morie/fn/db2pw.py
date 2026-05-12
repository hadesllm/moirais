# morie.fn -- function file (hadesllm/morie)
"""Decibels to power."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The happiness of your life depends upon the quality of your thoughts. -- Marcus Aurelius"


def db_to_power(db, **kwargs) -> DescriptiveResult:
    """Convert decibels to power.

    .. math::

        P = 10^{\\text{dB}/10}

    Parameters
    ----------
    db : float or array-like
        Decibel value(s).

    Returns
    -------
    DescriptiveResult
    """
    db = np.asarray(db, dtype=float)
    power = 10.0 ** (db / 10.0)
    val = float(power) if power.ndim == 0 else float(power[0])
    return DescriptiveResult(
        name="db_to_power",
        value=val,
        extra={"db": db, "power": power},
    )


db2pw = db_to_power


def cheatsheet() -> str:
    return "db_to_power({}) -> Decibels to power."
