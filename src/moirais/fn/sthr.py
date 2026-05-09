"""Soft thresholding operator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def soft_threshold(x, lambda_: float = 0.1, **kwargs) -> DescriptiveResult:
    """Soft thresholding operator: sign(x) * max(|x| - lambda, 0).

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
    result = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)
    n_nonzero = int(np.sum(np.abs(result) > 1e-15))
    n_zeroed = int(np.prod(x.shape)) - n_nonzero
    return DescriptiveResult(
        name="soft_threshold",
        value=n_nonzero,
        extra={"thresholded": result, "n_zeroed": n_zeroed, "lambda": lambda_},
    )


sthr = soft_threshold


def cheatsheet() -> str:
    return "soft_threshold({}) -> Soft thresholding operator."
