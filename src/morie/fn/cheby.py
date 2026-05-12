# morie.fn -- function file (hadesllm/morie)
"""Chebyshev (L-infinity) distance. 'Luck is what happens when preparation meets opportunity. -- Seneca' -- Ahsoka Tano"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def chebyshev_dist(a: np.ndarray, b: np.ndarray) -> DescriptiveResult:
    r"""
    Compute the Chebyshev distance (L-infinity norm) between two vectors.

    .. math::

        d_{\\infty}(a, b) = \\max_i |a_i - b_i|

    :param a: First vector.
    :type a: numpy.ndarray
    :param b: Second vector.
    :type b: numpy.ndarray
    :return: DescriptiveResult with distance value.
    :rtype: DescriptiveResult
    :raises ValueError: If a and b have different lengths.

    References
    ----------
    Chebyshev P.L. (1854). Theorie des mecanismes connus sous le nom de
    parallelogrammes. *Memoires de l'Academie Imperiale des Sciences de
    St.-Petersbourg*, 7, 539-568.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length: {len(a)} vs {len(b)}.")
    dist = float(np.max(np.abs(a - b)))
    return DescriptiveResult(
        name="chebyshev_distance",
        value=dist,
        extra={"distance": dist, "dim": len(a)},
    )


cheby = chebyshev_dist


def cheatsheet() -> str:
    return "chebyshev_dist({}) -> Chebyshev (L-infinity) distance. 'Luck is what happens when preparation meets opportunity. -- Seneca' -- Ahsoka T"
