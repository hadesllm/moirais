# morie.fn — function file (hadesllm/morie)
"""MLSMU6 single ALS iteration. 'Territory.' -- Yusuke, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def mlsmu6_single_iteration(X, Y, D):
    """One alternating least-squares step for unfolding.

    Parameters
    ----------
    X : array-like
        Respondent coordinates (n_resp x p).
    Y : array-like
        Stimulus coordinates (n_stim x p).
    D : array-like
        Target distance matrix (n_resp x n_stim).

    Returns
    -------
    DescriptiveResult
        value = (X_new, Y_new) tuple, extra has stress.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    n_resp, p = X.shape
    n_stim = Y.shape[0]

    D_hat = np.zeros((n_resp, n_stim))
    for i in range(n_resp):
        for j in range(n_stim):
            D_hat[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))

    X_new = X.copy()
    for i in range(n_resp):
        numer = np.zeros(p)
        denom = 0.0
        for j in range(n_stim):
            if D_hat[i, j] > 1e-12:
                ratio = D[i, j] / D_hat[i, j]
                numer += Y[j] + ratio * (X[i] - Y[j])
                denom += 1.0
        if denom > 0:
            X_new[i] = numer / denom

    Y_new = Y.copy()
    for j in range(n_stim):
        numer = np.zeros(p)
        denom = 0.0
        for i in range(n_resp):
            d_hat = np.sqrt(np.sum((X_new[i] - Y[j]) ** 2))
            if d_hat > 1e-12:
                ratio = D[i, j] / d_hat
                numer += X_new[i] + ratio * (Y[j] - X_new[i])
                denom += 1.0
        if denom > 0:
            Y_new[j] = numer / denom

    stress = 0.0
    for i in range(n_resp):
        for j in range(n_stim):
            d = np.sqrt(np.sum((X_new[i] - Y_new[j]) ** 2))
            stress += (D[i, j] - d) ** 2
    return DescriptiveResult(
        name="mlsmu6_single_iteration",
        value=(X_new, Y_new),
        extra={"stress": float(stress)},
    )


mlsit = mlsmu6_single_iteration


def cheatsheet() -> str:
    return "mlsmu6_single_iteration({}) -> MLSMU6 single ALS iteration. 'Territory.' -- Yusuke, Yu Yu H"
