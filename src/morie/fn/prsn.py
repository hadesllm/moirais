# morie.fn — function file (hadesllm/morie)
"""Pearson correlation coefficient."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful the mind of a child is."


def pearson_corr(x, y, **kwargs) -> DescriptiveResult:
    r"""Compute the Pearson product-moment correlation coefficient.

    .. math::

        r = \\frac{\\text{cov}(x, y)}{s_x \\cdot s_y}

    Parameters
    ----------
    x, y : array-like
        Input signals of equal length.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    return DescriptiveResult(
        name="pearson_corr",
        value=r,
        extra={"r": r, "r_squared": r**2, "n": len(x)},
    )


prsn = pearson_corr


def cheatsheet() -> str:
    return "pearson_corr({}) -> Pearson correlation coefficient."
