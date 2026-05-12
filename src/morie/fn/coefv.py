# morie.fn -- function file (hadesllm/morie)
"""Coefficient of variation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def coefficient_of_variation(x, **kwargs) -> DescriptiveResult:
    """Compute the coefficient of variation of signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    from morie._filters import coefficient_of_variation as _cv

    x = np.asarray(x, dtype=float)
    cv = _cv(x)
    return DescriptiveResult(
        name="coefficient_of_variation",
        value=cv,
        extra={"cv": cv},
    )


coefv = coefficient_of_variation


def cheatsheet() -> str:
    return "coefficient_of_variation({}) -> Coefficient of variation."
