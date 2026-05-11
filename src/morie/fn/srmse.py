"""Root mean squared error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def root_mean_squared_error(x, x_hat, **kwargs) -> DescriptiveResult:
    """Compute the root mean squared error between *x* and *x_hat*.

    .. math::

        \\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{n=0}^{N-1} (x(n) - \\hat{x}(n))^2}

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
    rmse = float(np.sqrt(mse))
    return DescriptiveResult(
        name="root_mean_squared_error",
        value=rmse,
        extra={"rmse": rmse, "mse": mse, "n": len(x)},
    )


srmse = root_mean_squared_error


def cheatsheet() -> str:
    return "root_mean_squared_error({}) -> Root mean squared error."
