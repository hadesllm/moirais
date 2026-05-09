"""SMACOF unfolding algorithm for rectangular dissimilarities."""

from __future__ import annotations

import numpy as np

from ._containers import MdsRes


def smacof_unfolding_basic(
    dissimilarities,
    n_dims: int = 2,
    *,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> MdsRes:
    """SMACOF rectangular unfolding (individuals x stimuli).

    Recovers ideal points and stimulus positions from a rectangular
    dissimilarity matrix (voters x alternatives).

    :param dissimilarities: (n_voters x n_stimuli) dissimilarity matrix.
    :param n_dims: Embedding dimensionality.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: MdsRes with concatenated [voters; stimuli] coordinates and stress.

    References
    ----------
    Armstrong (2014), Ch 4. De Leeuw & Heiser (1977) SMACOF.

    .. epigraph:: "The sleeper must awaken." -- Stilgar, Dune
    """
    Delta = np.asarray(dissimilarities, dtype=float)
    n_row, n_col = Delta.shape

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_row, n_dims))
    Y = rng.standard_normal((n_col, n_dims))

    def _dists(X, Y):
        diff = X[:, None, :] - Y[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1) + 1e-14)

    def _stress(X, Y):
        d = _dists(X, Y)
        return np.sqrt(((d - Delta) ** 2).sum() / max((Delta ** 2).sum(), 1e-14))

    stress = _stress(X, Y)
    for _ in range(max_iter):
        d = _dists(X, Y)
        ratio = np.where(d > 1e-14, Delta / d, 0.0)

        X_new = np.zeros_like(X)
        Y_new = np.zeros_like(Y)
        for i in range(n_row):
            X_new[i] = (ratio[i, :, None] * Y).sum(axis=0) / max(ratio[i].sum(), 1e-14)
        for j in range(n_col):
            Y_new[j] = (ratio[:, j, None] * X).sum(axis=0) / max(ratio[:, j].sum(), 1e-14)

        new_stress = _stress(X_new, Y_new)
        if abs(stress - new_stress) < tol:
            X, Y = X_new, Y_new
            stress = new_stress
            break
        X, Y = X_new, Y_new
        stress = new_stress

    coords = np.vstack([X, Y])
    return MdsRes(coordinates=coords, stress=stress, eigenvalues=np.array([]))


smcub = smacof_unfolding_basic


def cheatsheet() -> str:
    return "smacof_unfolding_basic({}) -> SMACOF rectangular unfolding."
