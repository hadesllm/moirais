# morie.fn — function file (hadesllm/morie)
"""Orbifold spectrum and fixed points."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def orbifold_spectrum(
    group_order: int = 3,
    twist: float = 1.0,
    d: int = 6,
) -> DescriptiveResult:
    r"""Compute orbifold fixed point count and twisted sector spectrum.

    For a :math:`\\mathbb{Z}_N` orbifold of a torus :math:`T^d`:

    - Fixed points: :math:`N^{d/2}` (for even d)
    - Twisted sectors: :math:`N - 1`

    :param group_order: Order N of the discrete group Z_N. Must be >= 2.
    :param twist: Twist parameter (fraction of 2pi).
    :param d: Number of compact dimensions (must be even).
    :return: DescriptiveResult with fixed point count and spectrum info.
    """
    if group_order < 2:
        raise ValueError(f"Group order must be >= 2, got {group_order}.")
    if d < 2 or d % 2 != 0:
        raise ValueError(f"Compact dimensions must be even >= 2, got {d}.")
    fixed_points = group_order ** (d // 2)
    twisted_sectors = group_order - 1
    twist_angles = [(k * twist * 2 * np.pi / group_order) for k in range(1, group_order)]
    return DescriptiveResult(
        name="orbifold_spectrum",
        value=float(fixed_points),
        extra={
            "group_order": group_order,
            "fixed_points": fixed_points,
            "twisted_sectors": twisted_sectors,
            "twist_angles": twist_angles,
            "compact_dimensions": d,
        },
    )


def cheatsheet() -> str:
    return "orbifold_spectrum(group_order, twist, d) -> orbifold fixed points"


orbfd = orbifold_spectrum
