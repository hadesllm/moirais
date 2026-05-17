# morie.fn -- function file (hadesllm/morie)
"""Compute the Minkowski distance (Lp norm) between two vectors."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def minkowski_dist(a: np.ndarray, b: np.ndarray, p: float = 2.0) -> DescriptiveResult:
    r"""
    Compute the Minkowski distance (Lp norm) between two vectors.

    .. math::

        d_p(a, b) = \\left(\\sum_i |a_i - b_i|^p\\right)^{1/p}

    Special cases: p=1 (Manhattan), p=2 (Euclidean).

    :param a: First vector.
    :type a: numpy.ndarray
    :param b: Second vector.
    :type b: numpy.ndarray
    :param p: Order of the norm. Must be >= 1. Default 2.
    :type p: float
    :return: DescriptiveResult with distance value.
    :rtype: DescriptiveResult
    :raises ValueError: If p < 1 or vectors have different lengths.

    References
    ----------
    Minkowski H. (1910). *Geometrie der Zahlen*. Teubner.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length: {len(a)} vs {len(b)}.")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}.")
    dist = float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))
    return DescriptiveResult(
        name="minkowski_distance",
        value=dist,
        extra={"distance": dist, "p": p, "dim": len(a)},
    )


minkw = minkowski_dist
