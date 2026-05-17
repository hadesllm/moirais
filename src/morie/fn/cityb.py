# morie.fn -- function file (hadesllm/morie)
"""City-block (Manhattan) distance."""

from __future__ import annotations

from ._containers import DescriptiveResult


def city_block_distance(x, y) -> DescriptiveResult:
    """Compute Manhattan / city-block distance.

    :param x: First point.
    :param y: Second point.
    :return: DescriptiveResult with distance.

    .. epigraph:: No man ever steps in the same river twice. -- Heraclitus
    """
    import numpy as np

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    d = float(np.sum(np.abs(x - y)))
    return DescriptiveResult(
        name="city_block_distance",
        value=d,
        extra={"n_dims": len(x)},
    )


cityb = city_block_distance


def cheatsheet() -> str:
    return "city_block_distance({}) -> City-block (Manhattan) distance."
