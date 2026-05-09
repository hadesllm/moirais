# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Estimation bias analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def bias_error(y_true, y_pred, **kwargs) -> DescriptiveResult:
    """Compute estimation bias.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted/estimated values.

    Returns
    -------
    DescriptiveResult
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    bias = float(np.mean(yp - yt))
    abs_bias = float(np.abs(bias))
    rel_bias = float(abs_bias / (np.abs(np.mean(yt)) + 1e-15))
    return DescriptiveResult(
        name="bias_error",
        value=bias,
        extra={
            "bias": bias,
            "abs_bias": abs_bias,
            "rel_bias": rel_bias,
            "mean_true": float(np.mean(yt)),
            "mean_pred": float(np.mean(yp)),
        },
    )


bserr = bias_error


def cheatsheet() -> str:
    return "bias_error({}) -> Estimation bias analysis."
