# moirais.fn — function file (hadesllm/moirais)
"""Kaiming (He) weight initialization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kaiming_init(
    fan_in: int,
    fan_out: int,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    seed: int | None = None,
) -> DescriptiveResult:
    """Generate Kaiming (He) initialized weight matrix.

    :math:`W \\sim \\mathcal{N}(0, \\sqrt{2 / \\text{fan}})` for ReLU.

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param mode: 'fan_in' or 'fan_out'.
    :param nonlinearity: 'relu' (gain=sqrt(2)) or 'linear' (gain=1).
    :param seed: Random seed.
    :return: DescriptiveResult with weight matrix in ``extra['weights']``.
    """
    rng = np.random.default_rng(seed)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / np.sqrt(fan)
    weights = rng.normal(0.0, std, size=(fan_out, fan_in))

    return DescriptiveResult(
        name="kaiming_init",
        value=float(std),
        extra={"weights": weights, "fan_in": fan_in, "fan_out": fan_out, "mode": mode, "gain": float(gain)},
    )


def cheatsheet() -> str:
    return "kaiming_init(fan_in, fan_out) -> He initialization weights"


kaimg = kaiming_init
