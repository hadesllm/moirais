# morie.fn — function file (hadesllm/morie)
"""Dynamic range."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Wars not make one great."


def dynamic_range(x, **kwargs) -> DescriptiveResult:
    r"""Compute the dynamic range in dB.

    .. math::

        DR = 20 \\cdot \\log_{10}\\left(\\frac{\\max|x|}{\\min_{>0}|x|}\\right)

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    abs_x = np.abs(x)
    max_val = np.max(abs_x)
    pos = abs_x[abs_x > 0]
    if len(pos) == 0 or max_val == 0:
        raise ValueError("Signal is all zeros; dynamic range is undefined.")
    min_val = np.min(pos)
    dr_db = 20.0 * np.log10(max_val / min_val)
    return DescriptiveResult(
        name="dynamic_range",
        value=float(dr_db),
        extra={"max_abs": float(max_val), "min_abs": float(min_val), "dr_db": float(dr_db)},
    )


dynrg = dynamic_range


def cheatsheet() -> str:
    return "dynamic_range({}) -> Dynamic range."
