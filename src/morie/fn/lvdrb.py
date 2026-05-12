# morie.fn -- function file (hadesllm/morie)
"""Levinson-Durbin recursion for AR model from ACF."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def levinson_durbin_fn(acf: np.ndarray, order: int = 4) -> DescriptiveResult:
    r"""Levinson-Durbin recursion: solve Toeplitz system from autocorrelation.

    Given autocorrelation sequence :math:`r(0), r(1), \\ldots, r(p)`,
    recursively solves for AR coefficients and prediction error.

    :param acf: Autocorrelation sequence r(0)...r(order).
    :param order: AR model order (default 4).
    :return: DescriptiveResult with AR coefficients and prediction error variance.
    """
    from morie._armodel import levinson_durbin

    acf = np.asarray(acf, dtype=float).ravel()
    if len(acf) < order + 1:
        raise ValueError("ACF must have at least order+1 values")
    a, sigma2 = levinson_durbin(acf, order)
    return DescriptiveResult(
        name="levinson_durbin",
        value=float(sigma2),
        extra={"coefficients": a, "sigma2": float(sigma2), "order": order},
    )


lvdrb = levinson_durbin_fn


def cheatsheet() -> str:
    return "levinson_durbin_fn({}) -> Levinson-Durbin recursion for AR model from ACF."
