"""Weighted Euclidean spatial model with salience weights."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def weighted_euclidean_model(ideal_points, alternatives, salience) -> DescriptiveResult:
    """Weighted Euclidean spatial model: U = -sum_k w_k*(x_ik - z_jk)^2.

    Dimension-specific salience weights allow different issue importance
    for each voter.

    :param ideal_points: (n_voters x n_dims) voter ideal points.
    :param alternatives: (n_alternatives x n_dims) alternative positions.
    :param salience: (n_voters x n_dims) salience weights per dimension.
    :return: DescriptiveResult with utility matrix and predicted choices.

    References
    ----------
    Armstrong (2014), Ch 5.

    .. epigraph:: I think, therefore I am. -- Rene Descartes
    """
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(alternatives, dtype=float)
    W = np.asarray(salience, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if W.ndim == 1:
        W = np.tile(W, (X.shape[0], 1))

    diff_sq = (X[:, None, :] - Z[None, :, :]) ** 2
    utility = -(W[:, None, :] * diff_sq).sum(axis=-1)
    predicted = np.argmax(utility, axis=1)

    return DescriptiveResult(
        name="weighted_euclidean_model",
        value={"utility": utility, "predicted_choice": predicted},
        extra={"n_voters": X.shape[0], "n_alternatives": Z.shape[0]},
    )


wghtm = weighted_euclidean_model


def cheatsheet() -> str:
    return "weighted_euclidean_model({}) -> Weighted Euclidean spatial model."
