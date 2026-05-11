"""Mean squared error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def mean_squared_error(x, x_hat, **kwargs) -> DescriptiveResult:
    """Compute the mean squared error between *x* and *x_hat*.

    .. math::

        \\text{MSE} = \\frac{1}{N} \\sum_{n=0}^{N-1} (x(n) - \\hat{x}(n))^2

    Parameters
    ----------
    x : array-like
        Reference signal.
    x_hat : array-like
        Estimated / reconstructed signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    mse = float(np.mean((x - x_hat) ** 2))
    return DescriptiveResult(
        name="mean_squared_error",
        value=mse,
        extra={"mse": mse, "n": len(x)},
    )


smse = mean_squared_error


def cheatsheet() -> str:
    return "mean_squared_error({}) -> Mean squared error."
