# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Canberra distance. 'Hope is like the sun.' -- Vice Admiral Holdo"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def canberra_dist(a: np.ndarray, b: np.ndarray) -> DescriptiveResult:
    r"""
    Compute the Canberra distance between two vectors.

    .. math::

        d_C(a, b) = \\sum_i \\frac{|a_i - b_i|}{|a_i| + |b_i|}

    Terms where both :math:`a_i` and :math:`b_i` are zero are skipped.

    :param a: First vector.
    :type a: numpy.ndarray
    :param b: Second vector.
    :type b: numpy.ndarray
    :return: DescriptiveResult with distance value.
    :rtype: DescriptiveResult
    :raises ValueError: If vectors have different lengths.

    References
    ----------
    Lance G.N. & Williams W.T. (1966). Computer programs for
    hierarchical polythetic classification. *Computer Journal*, 9(1),
    60-64.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length: {len(a)} vs {len(b)}.")
    denom = np.abs(a) + np.abs(b)
    mask = denom > 0
    dist = float(np.sum(np.abs(a[mask] - b[mask]) / denom[mask]))
    return DescriptiveResult(
        name="canberra_distance",
        value=dist,
        extra={"distance": dist, "dim": len(a)},
    )


canbs = canberra_dist


def cheatsheet() -> str:
    return "canberra_dist({}) -> Canberra distance. 'Hope is like the sun.' -- Vice Admiral H"
