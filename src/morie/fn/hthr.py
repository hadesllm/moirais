# morie.fn — function file (hadesllm/morie)
"""Hard thresholding operator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficulties strengthen the mind, as labor does the body. — Seneca"


def hard_threshold(x, lambda_: float = 0.1, **kwargs) -> DescriptiveResult:
    """Hard thresholding operator: x * (|x| > lambda).

    Parameters
    ----------
    x : array-like
        Input signal / coefficients.
    lambda_ : float
        Threshold level (default 0.1).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of non-zero entries after thresholding;
        ``extra`` has ``thresholded``, ``n_zeroed``, ``lambda``.
    """
    x = np.asarray(x, dtype=float)
    mask = np.abs(x) > lambda_
    result = x * mask
    n_nonzero = int(np.sum(mask))
    n_zeroed = int(np.prod(x.shape)) - n_nonzero
    return DescriptiveResult(
        name="hard_threshold",
        value=n_nonzero,
        extra={"thresholded": result, "n_zeroed": n_zeroed, "lambda": lambda_},
    )


hthr = hard_threshold


def cheatsheet() -> str:
    return "hard_threshold({}) -> Hard thresholding operator."
