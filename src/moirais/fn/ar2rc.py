# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AR coefficients to reflection coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def ar_to_reflection(ar_coeffs, **kwargs) -> DescriptiveResult:
    """Convert AR coefficients to reflection coefficients.

    Parameters
    ----------
    ar_coeffs : array-like
        AR coefficients (with leading 1).

    Returns
    -------
    DescriptiveResult
    """
    a = np.asarray(ar_coeffs, dtype=float)
    if abs(a[0]) > 0:
        a = a / a[0]
    a = a[1:].copy()
    p = len(a)
    rc = np.zeros(p)
    for i in range(p - 1, -1, -1):
        rc[i] = a[i]
        if abs(rc[i]) >= 1.0:
            break
        prev = a[:i].copy()
        denom = 1.0 - rc[i] ** 2
        if abs(denom) < 1e-15:
            break
        for j in range(i):
            a[j] = (prev[j] - rc[i] * prev[i - 1 - j]) / denom
    return DescriptiveResult(
        name="ar_to_reflection",
        value=float(rc[0]) if len(rc) > 0 else 0.0,
        extra={"rc": rc, "ar": ar_coeffs},
    )


ar2rc = ar_to_reflection


def cheatsheet() -> str:
    return "ar_to_reflection({}) -> AR coefficients to reflection coefficients."
