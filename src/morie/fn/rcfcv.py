# morie.fn — function file (hadesllm/morie)
"""Reflection coefficients to AR coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what we repeatedly do. Excellence is not an act, but a habit. — Aristotle"


def reflection_to_ar(rc, **kwargs) -> DescriptiveResult:
    """Convert reflection coefficients to AR coefficients (Levinson recursion).

    Parameters
    ----------
    rc : array-like
        Reflection coefficients [k1, k2, ...].

    Returns
    -------
    DescriptiveResult
    """
    rc = np.asarray(rc, dtype=float)
    p = len(rc)
    a = np.zeros(p)
    a[0] = rc[0]
    for i in range(1, p):
        prev = a[:i].copy()
        a[i] = rc[i]
        for j in range(i):
            a[j] = prev[j] + rc[i] * prev[i - 1 - j]
    ar = np.concatenate(([1.0], a))
    return DescriptiveResult(
        name="reflection_to_ar",
        value=float(ar[1]),
        extra={"ar": ar, "rc": rc},
    )


rcfcv = reflection_to_ar


def cheatsheet() -> str:
    return "reflection_to_ar({}) -> Reflection coefficients to AR coefficients."
