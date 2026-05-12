"""Unfolding model -- ideal point estimation from preference data."""

from __future__ import annotations

import numpy as np

from ._containers import MdsRes


def unfolding_model(
    preferences,
    n_dims: int = 2,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> MdsRes:
    """Metric unfolding: recover ideal points from rank-order preferences.

    Given an (n_voters x n_stimuli) matrix of preference rankings (1 = most
    preferred), converts to dissimilarities and applies alternating
    least-squares unfolding.

    :param preferences: (n_voters x n_stimuli) matrix of rankings.
    :param n_dims: Dimensionality of recovered space.
    :param max_iter: Maximum ALS iterations.
    :param tol: Convergence tolerance.
    :return: MdsRes with ideal points and stimulus coordinates.

    References
    ----------
    Armstrong (2014), Ch 4. Coombs (1964) unfolding theory.

    .. epigraph:: "I am no man." -- Eowyn, Lord of the Rings
    """
    R = np.asarray(preferences, dtype=float)
    n_row, n_col = R.shape

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_row, n_dims))
    Y = rng.standard_normal((n_col, n_dims))

    def _loss(X, Y):
        diff = X[:, None, :] - Y[None, :, :]
        d2 = (diff ** 2).sum(axis=-1)
        return ((d2 - R) ** 2).sum()

    prev_loss = _loss(X, Y)
    for _ in range(max_iter):
        for i in range(n_row):
            w = 1.0 / (np.abs(R[i]) + 1e-8)
            X[i] = (w[:, None] * Y).sum(axis=0) / w.sum()
        for j in range(n_col):
            w = 1.0 / (np.abs(R[:, j]) + 1e-8)
            Y[j] = (w[:, None] * X).sum(axis=0) / w.sum()
        cur_loss = _loss(X, Y)
        if abs(prev_loss - cur_loss) < tol:
            break
        prev_loss = cur_loss

    coords = np.vstack([X, Y])
    stress = np.sqrt(prev_loss / max((R ** 2).sum(), 1e-14))
    return MdsRes(coordinates=coords, stress=stress, eigenvalues=np.array([]))


unfld = unfolding_model


def cheatsheet() -> str:
    return "unfolding_model({}) -> Metric unfolding for ideal point estimation."
