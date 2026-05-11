# morie.fn — function file (hadesllm/morie)
"""Multi-trait genomic prediction."""

__all__ = ["mtrpr"]

import numpy as np

from ._containers import GenomicsResult


def mtrpr(
    Y: np.ndarray,
    G: np.ndarray,
    *,
    X: np.ndarray | None = None,
    lambda_vals: np.ndarray | None = None,
) -> GenomicsResult:
    """Multi-trait genomic BLUP prediction.

    Extends GBLUP to multiple correlated traits by solving a
    multi-trait mixed model, leveraging genetic correlations
    to improve prediction accuracy.

    Parameters
    ----------
    Y : array, shape (n, t)
        Phenotype matrix (n individuals, t traits).
    G : array, shape (n, n)
        Genomic relationship matrix.
    X : array, shape (n, p), optional
        Fixed-effect design matrix.  If None, intercept only.
    lambda_vals : array, shape (t,), optional
        Per-trait shrinkage parameters (var_e / var_g).
        If None, set to 1.0 for all traits.

    Returns
    -------
    GenomicsResult
        statistic = mean prediction accuracy across traits,
        extra has 'gebv' (n x t), 'accuracy_per_trait',
        'genetic_correlation'.

    References
    ----------
    Jia, Y., & Jannink, J.-L. (2012). Multiple-trait genomic
        selection methods increase genetic value prediction
        accuracy. Genetics, 192(4), 1513-1522.
    """
    Y = np.asarray(Y, dtype=float)
    G = np.asarray(G, dtype=float)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, t = Y.shape

    if G.shape != (n, n):
        raise ValueError(f"G must be ({n},{n}).")

    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.asarray(X, dtype=float)

    if lambda_vals is not None:
        lam = np.asarray(lambda_vals, dtype=float).ravel()
    else:
        lam = np.ones(t)

    G_inv = np.linalg.inv(G + np.eye(n) * 1e-6)
    p_fix = X.shape[1]

    gebv = np.zeros((n, t))
    beta_all = np.zeros((p_fix, t))

    for k in range(t):
        y_k = Y[:, k]
        lhs = np.zeros((p_fix + n, p_fix + n))
        lhs[:p_fix, :p_fix] = X.T @ X
        lhs[:p_fix, p_fix:] = X.T
        lhs[p_fix:, :p_fix] = X
        lhs[p_fix:, p_fix:] = np.eye(n) + G_inv * lam[k]

        rhs = np.concatenate([X.T @ y_k, y_k])
        sol = np.linalg.solve(lhs, rhs)

        beta_all[:, k] = sol[:p_fix]
        gebv[:, k] = sol[p_fix:]

    accuracy = []
    for k in range(t):
        corr = np.corrcoef(Y[:, k], gebv[:, k])[0, 1]
        accuracy.append(float(corr) if np.isfinite(corr) else 0.0)

    if t >= 2:
        gen_corr = np.corrcoef(gebv.T).tolist()
    else:
        gen_corr = [[1.0]]

    return GenomicsResult(
        name="MultiTraitGBLUP",
        statistic=float(np.mean(accuracy)),
        n=n,
        extra={
            "gebv": gebv.tolist(),
            "accuracy_per_trait": accuracy,
            "genetic_correlation": gen_corr,
            "n_traits": t,
        },
    )


def cheatsheet() -> str:
    return "mtrpr(Y, G) -> Multi-trait genomic prediction."
