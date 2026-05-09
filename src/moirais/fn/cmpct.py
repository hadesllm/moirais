# moirais.fn — function file (hadesllm/moirais)
"""Torus compactification moduli."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def compactify_torus(
    radii: list[float] | np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute moduli space properties of T^n toroidal compactification.

    For a rectangular torus :math:`T^n` with radii :math:`R_1, \\ldots, R_n`:

    - Volume: :math:`V = (2\\pi)^n \\prod_i R_i`
    - Moduli count: :math:`n^2` (metric) for a general torus

    :param radii: Compactification radii. Defaults to [1.0, 1.0] (T^2).
    :return: DescriptiveResult with volume, dimension, moduli count.
    """
    if radii is None:
        radii = [1.0, 1.0]
    radii = np.asarray(radii, dtype=float)
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")
    n = len(radii)
    volume = (2 * np.pi) ** n * np.prod(radii)
    moduli_count = n * n
    return DescriptiveResult(
        name="compactify_torus",
        value=float(volume),
        extra={
            "radii": radii,
            "dimension": n,
            "volume": float(volume),
            "moduli_count": moduli_count,
        },
    )


def cheatsheet() -> str:
    return "compactify_torus(radii) -> T^n compactification moduli"


cmpct = compactify_torus
