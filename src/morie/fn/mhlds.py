# morie.fn — function file (hadesllm/morie)
"""Mahalanobis distance computation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mhlds_fn(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> DescriptiveResult:
    """Compute Mahalanobis distance from a point to a distribution.

    :param x: 1-D observation vector.
    :param mu: 1-D mean vector of the distribution.
    :param cov: 2-D covariance matrix.
    :return: DescriptiveResult with distance value.
    """
    from morie._classify import mahalanobis_distance

    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    dist = mahalanobis_distance(x, mu, cov)
    return DescriptiveResult(
        name="mahalanobis",
        value=dist,
        extra={"distance": dist},
    )


mhlds = mhlds_fn


def cheatsheet() -> str:
    return "mhlds_fn({}) -> Mahalanobis distance computation."
