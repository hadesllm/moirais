# morie.fn — function file (hadesllm/morie)
"""Riemannian covariance estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rcov_fn(X: np.ndarray, metric: str = "riemann") -> DescriptiveResult:
    """Compute covariance matrix with optional Riemannian metric.

    :param X: 2-D array (channels x samples).
    :param metric: Metric type ('riemann' or 'logeuclid', default 'riemann').
    :return: DescriptiveResult with matrix dimension and covariance.
    """
    from morie._decompose import riemannian_covariance

    X = np.asarray(X, dtype=float)
    C = riemannian_covariance(X, metric=metric)
    return DescriptiveResult(
        name="riemannian_cov",
        value=C.shape[0],
        extra={"covariance": C, "metric": metric},
    )


rcov = rcov_fn


def cheatsheet() -> str:
    return "rcov_fn({}) -> Riemannian covariance estimation."
