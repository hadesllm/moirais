# morie.fn — function file (hadesllm/morie)
"""Genomic BLUP — Henderson mixed model equations."""

__all__ = ["gblup"]

import numpy as np

from ._containers import GenomicsResult


def gblup(
    y: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    *,
    lambda_val: float | None = None,
    var_e: float = 1.0,
    var_g: float = 1.0,
) -> GenomicsResult:
    """Genomic BLUP via Henderson mixed model equations.

    Solves the MME system::

        [X'X        X'Z      ] [b_hat]   [X'y]
        [Z'X   Z'Z + G^{-1}λ] [g_hat] = [Z'y]

    where λ = var_e / var_g, Z = I (all individuals genotyped),
    and G is the genomic relationship matrix.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    X : array, shape (n, p)
        Fixed-effect design matrix (include intercept column).
    G : array, shape (n, n)
        Genomic relationship matrix (positive definite).
    lambda_val : float, optional
        Ratio var_e / var_g.  Computed from var_e, var_g if not given.
    var_e : float
        Residual variance (default 1.0).
    var_g : float
        Genetic variance (default 1.0).

    Returns
    -------
    GenomicsResult
        statistic = mean |GEBV|, extra has 'beta' (fixed effects)
        and 'gebv' (genomic estimated breeding values).

    References
    ----------
    VanRaden, P. M. (2008). Efficient methods to compute genomic
        predictions. J. Dairy Sci., 91(11), 4414-4423.
    Henderson, C. R. (1984). Applications of Linear Models in Animal
        Breeding. Univ. Guelph Press.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    G = np.asarray(G, dtype=float)
    n = len(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if G.shape != (n, n):
        raise ValueError(f"G must be ({n},{n}), got {G.shape}.")
    if X.shape[0] != n:
        raise ValueError("X and y must have the same number of rows.")

    lam = lambda_val if lambda_val is not None else var_e / var_g

    Z = np.eye(n)
    G_inv = np.linalg.inv(G + np.eye(n) * 1e-6)

    p = X.shape[1]
    lhs = np.zeros((p + n, p + n))
    lhs[:p, :p] = X.T @ X
    lhs[:p, p:] = X.T @ Z
    lhs[p:, :p] = Z.T @ X
    lhs[p:, p:] = Z.T @ Z + G_inv * lam

    rhs = np.concatenate([X.T @ y, Z.T @ y])
    sol = np.linalg.solve(lhs, rhs)

    beta = sol[:p]
    gebv = sol[p:]

    return GenomicsResult(
        name="GBLUP",
        statistic=float(np.mean(np.abs(gebv))),
        n=n,
        extra={"beta": beta.tolist(), "gebv": gebv.tolist()},
    )


def cheatsheet() -> str:
    return "gblup(y, X, G) -> Genomic BLUP via Henderson mixed model equations."
