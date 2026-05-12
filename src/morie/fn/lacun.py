# morie.fn -- function file (hadesllm/morie)
"""Lacunarity analysis for fractal texture characterization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lacunarity_fn(
    x: np.ndarray,
    box_sizes: np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute lacunarity across box sizes for a 1-D signal.

    :param x: 1-D input signal.
    :param box_sizes: Array of box sizes; auto-generated if None.
    :return: DescriptiveResult with lacunarity values per box size.
    """
    from morie._adaptive import lacunarity

    x = np.asarray(x, dtype=float).ravel()
    lac, sizes = lacunarity(x, box_sizes=box_sizes)
    return DescriptiveResult(
        name="lacunarity",
        value=float(np.mean(lac)),
        extra={"lacunarity": lac, "box_sizes": sizes},
    )


lacun = lacunarity_fn


def cheatsheet() -> str:
    return "lacunarity_fn({}) -> Lacunarity analysis for fractal texture characterization."
