# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Normalize AR coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies, with thunderous applause."


def ar_normalize(ar_coeffs, **kwargs) -> DescriptiveResult:
    """Normalize AR coefficients so that a[0] = 1.

    Parameters
    ----------
    ar_coeffs : array-like
        AR coefficient vector.

    Returns
    -------
    DescriptiveResult
    """
    ar = np.asarray(ar_coeffs, dtype=float)
    if ar[0] == 0:
        raise ValueError("Leading coefficient is zero; cannot normalize.")
    normed = ar / ar[0]
    return DescriptiveResult(
        name="ar_normalize",
        value=float(normed[0]),
        extra={"ar_normalized": normed, "original": ar},
    )


arnrm = ar_normalize


def cheatsheet() -> str:
    return "ar_normalize({}) -> Normalize AR coefficients."
