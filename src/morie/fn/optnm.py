# morie.fn — function file (hadesllm/morie)
"""Nelder-Mead MDS optimisation. 'Yare yare daze.' -- Jotaro, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def nelder_mead_mds(D, n_dims=2, seed=42, maxiter=2000):
    """Optimize MDS coordinates via Nelder-Mead (scipy).

    Parameters
    ----------
    D : array-like
        Target distance matrix.
    n_dims : int
        Embedding dimensionality.
    seed : int
        Random seed for initial config.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        value = coordinate matrix, extra has final stress.
    """
    import numpy as np
    from scipy.optimize import minimize

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n * n_dims)

    def _stress(flat):
        X = flat.reshape(n, n_dims)
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d_hat = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                s += (D[i, j] - d_hat) ** 2
        return s

    res = minimize(_stress, x0, method="Nelder-Mead", options={"maxiter": maxiter})
    X_opt = res.x.reshape(n, n_dims)
    return DescriptiveResult(
        name="nelder_mead_mds", value=X_opt, extra={"stress": float(res.fun), "converged": res.success}
    )


optnm = nelder_mead_mds


def cheatsheet() -> str:
    return "nelder_mead_mds({}) -> Nelder-Mead MDS optimisation. 'Yare yare daze.' -- Jotaro, J"
