# morie.fn — function file (hadesllm/morie)
"""Separating hyperplane between two groups."""

from __future__ import annotations

from ._containers import DescriptiveResult


def separating_hyperplane(X, labels) -> DescriptiveResult:
    """Find separating hyperplane normal vector and midpoint between two groups.

    :param X: Data matrix (n_samples x n_dims).
    :param labels: Binary labels (0/1).
    :return: DescriptiveResult with normal vector and midpoint.

    .. epigraph:: "I am justice!" -- Light Yagami, Death Note
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    g0 = X[labels == 0]
    g1 = X[labels == 1]
    mu0 = g0.mean(axis=0)
    mu1 = g1.mean(axis=0)
    normal = mu1 - mu0
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    midpoint = (mu0 + mu1) / 2.0
    return DescriptiveResult(
        name="separating_hyperplane",
        value=float(norm),
        extra={"normal": normal.tolist(), "midpoint": midpoint.tolist()},
    )


sephy = separating_hyperplane


def cheatsheet() -> str:
    return "separating_hyperplane({}) -> Separating hyperplane between two groups."
