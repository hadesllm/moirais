# morie.fn — function file (hadesllm/morie)
"""Mahalanobis distance. 'Judge me by my size, do you?'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mahalanobis_dist(x: np.ndarray, data: np.ndarray) -> DescriptiveResult:
    r"""
    Compute the Mahalanobis distance of a point from a dataset.

    .. math::

        D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}

    :param x: 1-D point vector of length p.
    :type x: numpy.ndarray
    :param data: (n, p) reference dataset to estimate mean and covariance.
    :type data: numpy.ndarray
    :return: DescriptiveResult with Mahalanobis distance.
    :rtype: DescriptiveResult
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Mahalanobis P.C. (1936). On the generalized distance in statistics.
    *Proceedings of the National Institute of Sciences of India*, 2(1),
    49-55.
    """
    x = np.asarray(x, dtype=float).ravel()
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D.")
    if x.shape[0] != data.shape[1]:
        raise ValueError(f"x has {x.shape[0]} features but data has {data.shape[1]}.")
    mu = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    dist = float(np.sqrt(diff @ cov_inv @ diff))
    return DescriptiveResult(
        name="mahalanobis_distance",
        value=dist,
        extra={"distance": dist, "mean": mu, "p": len(x)},
    )


mahal = mahalanobis_dist


def cheatsheet() -> str:
    return "mahalanobis_dist({}) -> Mahalanobis distance. 'Judge me by my size, do you?'"
