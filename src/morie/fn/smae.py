"""Mean absolute error."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Wars not make one great."


def mean_absolute_error(x, x_hat, **kwargs) -> DescriptiveResult:
    """Compute the mean absolute error between *x* and *x_hat*.

    .. math::

        \\text{MAE} = \\frac{1}{N} \\sum_{n=0}^{N-1} |x(n) - \\hat{x}(n)|

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
    mae = float(np.mean(np.abs(x - x_hat)))
    return DescriptiveResult(
        name="mean_absolute_error",
        value=mae,
        extra={"mae": mae, "n": len(x)},
    )


smae = mean_absolute_error


def cheatsheet() -> str:
    return "mean_absolute_error({}) -> Mean absolute error."
