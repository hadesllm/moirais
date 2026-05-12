"""Xavier/Glorot weight initialization. 'By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. -- Socrates' -- Various"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def xavier_init(
    fan_in: int,
    fan_out: int,
    seed: int = 42,
    uniform: bool = True,
) -> DescriptiveResult:
    r"""
    Generate Xavier/Glorot weight initialization matrix.

    For uniform distribution:

    .. math::

        W \\sim U\\left[-\\sqrt{\\frac{6}{n_{in} + n_{out}}},
                        \\sqrt{\\frac{6}{n_{in} + n_{out}}}\\right]

    For normal distribution:

    .. math::

        W \\sim \\mathcal{N}\\left(0, \\frac{2}{n_{in} + n_{out}}\\right)

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param seed: Random seed. Default 42.
    :param uniform: Use uniform (True) or normal (False) init. Default True.
    :return: DescriptiveResult with weight matrix and statistics.
    :raises ValueError: If fan_in or fan_out <= 0.

    References
    ----------
    Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of
    training deep feedforward neural networks. *AISTATS*.
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(f"fan_in and fan_out must be > 0, got {fan_in}, {fan_out}.")

    rng = np.random.default_rng(seed)

    if uniform:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
    else:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        W = rng.normal(0, std, size=(fan_in, fan_out))

    return DescriptiveResult(
        name="Xavier Initialization",
        value=float(np.std(W)),
        extra={
            "weights": W,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "mean": float(np.mean(W)),
            "std": float(np.std(W)),
            "shape": (fan_in, fan_out),
            "method": "uniform" if uniform else "normal",
        },
    )


short = xavier_init


def cheatsheet() -> str:
    return "Nature does not hurry, yet everything is accomplished. -- Lao Tzu"
