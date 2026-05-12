# morie.fn — function file (hadesllm/morie)
"""A journey of a thousand miles begins with a single step. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def relu(
    x: np.ndarray,
    leaky: float = 0.0,
) -> DescriptiveResult:
    r"""
    Compute ReLU or Leaky ReLU activation function.

    .. math::

        \\text{ReLU}(x) = \\max(0, x)

    .. math::

        \\text{LeakyReLU}(x) = \\max(\\alpha x, x)

    :param x: Input array.
    :param leaky: Negative slope coefficient. Default 0 (standard ReLU).
    :return: DescriptiveResult with activated output and gradient.
    :raises ValueError: If leaky coefficient is negative.

    References
    ----------
    Nair, V., & Hinton, G. E. (2010). Rectified linear units improve
    restricted Boltzmann machines. *ICML*.
    """
    if leaky < 0:
        raise ValueError(f"leaky slope must be >= 0, got {leaky}.")

    x_arr = np.asarray(x, dtype=np.float64)
    output = np.where(x_arr > 0, x_arr, leaky * x_arr)
    grad = np.where(x_arr > 0, 1.0, leaky)

    frac_active = float(np.mean(x_arr > 0))

    return DescriptiveResult(
        name="ReLU" if leaky == 0 else f"LeakyReLU(alpha={leaky})",
        value=float(np.mean(output)),
        extra={
            "output": output,
            "gradient": grad,
            "fraction_active": frac_active,
            "leaky": leaky,
        },
    )


short = relu


def cheatsheet() -> str:
    return "A journey of a thousand miles begins with a single step. — Lao Tzu"
