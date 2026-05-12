"""Xavier (Glorot) weight initialization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def xavier_init(
    fan_in: int,
    fan_out: int,
    distribution: str = "normal",
    seed: int | None = None,
) -> DescriptiveResult:
    r"""Generate Xavier (Glorot) initialized weight matrix.

    Normal: :math:`W \\sim \\mathcal{N}(0, \\sqrt{2/(\\text{fan\\_in}+\\text{fan\\_out})})`.
    Uniform: :math:`W \\sim U[-a, a]` where :math:`a = \\sqrt{6/(\\text{fan\\_in}+\\text{fan\\_out})}`.

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param distribution: 'normal' or 'uniform'.
    :param seed: Random seed.
    :return: DescriptiveResult with weight matrix in ``extra['weights']``.
    """
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights = rng.normal(0.0, std, size=(fan_out, fan_in))
    elif distribution == "uniform":
        a = np.sqrt(6.0 / (fan_in + fan_out))
        weights = rng.uniform(-a, a, size=(fan_out, fan_in))
    else:
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution}")

    return DescriptiveResult(
        name="xavier_init",
        value=float(np.std(weights)),
        extra={"weights": weights, "fan_in": fan_in, "fan_out": fan_out, "distribution": distribution},
    )


def cheatsheet() -> str:
    return "xavier_init(fan_in, fan_out) -> Glorot initialization weights"


xvrig = xavier_init
