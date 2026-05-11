# morie.fn — function file (hadesllm/morie)
"""Radial Basis Function network."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rbfnn_fn(X_train: np.ndarray, y_train: np.ndarray, n_centers: int = 10, sigma: float = 1.0) -> DescriptiveResult:
    """Train an RBF network for classification/regression.

    :param X_train: Training features (samples x features).
    :param y_train: Training targets.
    :param n_centers: Number of RBF centers (default 10).
    :param sigma: RBF width parameter (default 1.0).
    :return: DescriptiveResult with weight count and weights/centers/sigma.
    """
    from morie._classify import rbf_network

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    weights, centers, sig = rbf_network(X_train, y_train, n_centers=n_centers, sigma=sigma)
    return DescriptiveResult(
        name="rbf_network",
        value=len(weights),
        extra={"weights": weights, "centers": centers, "sigma": sig},
    )


rbfnn = rbfnn_fn


def cheatsheet() -> str:
    return "rbfnn_fn({}) -> Radial Basis Function network."
