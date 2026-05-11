# morie.fn — function file (hadesllm/morie)
"""MSE = Bias^2 + Variance decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Much to learn you still have."


def mse_variance_bias(y_true, y_pred, **kwargs) -> DescriptiveResult:
    """Decompose MSE into bias squared and variance.

    MSE = Bias^2 + Variance, where Bias = mean(y_pred) - mean(y_true)
    and Variance = var(y_pred - y_true).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    DescriptiveResult
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    errors = yp - yt
    bias = float(np.mean(errors))
    variance = float(np.var(errors))
    mse = float(np.mean(errors**2))
    return DescriptiveResult(
        name="mse_variance_bias",
        value=mse,
        extra={
            "mse": mse,
            "bias": bias,
            "bias_sq": bias**2,
            "variance": variance,
        },
    )


msevr = mse_variance_bias


def cheatsheet() -> str:
    return "mse_variance_bias({}) -> MSE = Bias^2 + Variance decomposition."
