# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Batch normalization. 'Truly wonderful, the mind of a child is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def batch_norm(
    x: np.ndarray,
    gamma: float = 1.0,
    beta: float = 0.0,
    eps: float = 1e-5,
) -> DescriptiveResult:
    """
    Apply batch normalization to an input array.

    .. math::

        \\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}

    .. math::

        y = \\gamma \\hat{x} + \\beta

    :param x: Input array (1D or 2D, normalized over axis 0).
    :param gamma: Scale parameter. Default 1.0.
    :param beta: Shift parameter. Default 0.0.
    :param eps: Small constant for numerical stability. Default 1e-5.
    :return: DescriptiveResult with normalized output and statistics.

    References
    ----------
    Ioffe, S., & Szegedy, C. (2015). Batch normalization: accelerating
    deep network training by reducing internal covariate shift. *ICML*.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    mu = np.mean(x_arr, axis=0)
    var = np.var(x_arr, axis=0)
    x_hat = (x_arr - mu) / np.sqrt(var + eps)
    output = gamma * x_hat + beta

    return DescriptiveResult(
        name="Batch Normalization",
        value=float(np.mean(output)),
        extra={
            "output": output,
            "x_hat": x_hat,
            "mean": mu if np.ndim(mu) == 0 else mu,
            "variance": var if np.ndim(var) == 0 else var,
            "gamma": gamma,
            "beta": beta,
        },
    )


short = batch_norm


def cheatsheet() -> str:
    return "batch_norm({}) -> Batch normalization. 'Truly wonderful, the mind of a child i"
