# morie.fn -- function file (hadesllm/morie)
"""Reflection (PARCOR) coefficients from AR model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def reflection_coeff_fn(x: np.ndarray, order: int = 10) -> DescriptiveResult:
    """Compute reflection coefficients from AR model coefficients.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :return: DescriptiveResult with reflection coefficients in extra.
    """
    from morie._armodel import ar_yule_walker, reflection_coefficients

    x = np.asarray(x, dtype=float).ravel()
    a, _ = ar_yule_walker(x, order=order)
    k = reflection_coefficients(a)
    return DescriptiveResult(
        name="reflection_coefficients",
        value=None,
        extra={"reflection_coefficients": k},
    )


refco = reflection_coeff_fn


def cheatsheet() -> str:
    return "reflection_coeff_fn({}) -> Reflection (PARCOR) coefficients from AR model."
