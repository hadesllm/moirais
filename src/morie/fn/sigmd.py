"""Sigmoid activation with gradient. 'Luminous beings are we, not this crude matter.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sigmoid(
    x: np.ndarray,
) -> DescriptiveResult:
    r"""
    Compute the sigmoid activation function and its gradient.

    .. math::

        \\sigma(x) = \\frac{1}{1 + e^{-x}}

    .. math::

        \\sigma'(x) = \\sigma(x)(1 - \\sigma(x))

    :param x: Input array.
    :return: DescriptiveResult with sigmoid output and gradient.

    References
    ----------
, J., & Moraga, C. (1995). The influence of the sigmoid function
    parameters on the speed of backpropagation learning. *IWANN*.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    output = 1.0 / (1.0 + np.exp(-np.clip(x_arr, -500, 500)))
    grad = output * (1.0 - output)

    return DescriptiveResult(
        name="Sigmoid",
        value=float(np.mean(output)),
        extra={
            "output": output,
            "gradient": grad,
            "mean_output": float(np.mean(output)),
            "mean_gradient": float(np.mean(grad)),
        },
    )


short = sigmoid


def cheatsheet() -> str:
    return "sigmoid({}) -> Sigmoid activation with gradient. 'Luminous beings are we, n"
