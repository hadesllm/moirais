# morie.fn -- function file (hadesllm/morie)
"""Even-odd signal decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def even_odd_decompose(x) -> DescriptiveResult:
    """Decompose signal *x* into even and odd components.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    from morie._filters import even_odd_decompose as _eo

    x = np.asarray(x, dtype=float)
    x_even, x_odd = _eo(x)
    return DescriptiveResult(
        name="even_odd_decompose",
        value=len(x),
        extra={"even": x_even, "odd": x_odd},
    )


evnod = even_odd_decompose


def cheatsheet() -> str:
    return "even_odd_decompose({}) -> Even-odd signal decomposition."
