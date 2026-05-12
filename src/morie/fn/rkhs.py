# morie.fn -- function file (hadesllm/morie)
"""RKHS regression for genomic prediction."""

__all__ = ["rkhs"]

import numpy as np

from ._containers import GenomicsResult


def rkhs(
    y: np.ndarray,
    K: np.ndarray,
    *,
    lambda_val: float = 1.0,
) -> GenomicsResult:
    """Reproducing Kernel Hilbert Space regression for genomic prediction.

    Solves: y_hat = K (K + lambda * I)^{-1} y

    This is kernel ridge regression with a genomic kernel matrix.
    Equivalent to GBLUP when K is the genomic relationship matrix.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    K : array, shape (n, n)
        Kernel (Gram) matrix.  Common choices: Gaussian kernel of marker
        genotypes, or the genomic relationship matrix G.
    lambda_val : float
        Regularization parameter (default 1.0).

    Returns
    -------
    GenomicsResult
        statistic = correlation(y, y_hat),
        extra has 'fitted' and 'alpha' (dual weights).

    References
    ----------
    Gianola, D., & van Kaam, J. B. (2008). Reproducing kernel Hilbert
        spaces regression methods for genomic assisted prediction of
        quantitative traits. Genetics, 178(4), 2289-2303.
    """
    y = np.asarray(y, dtype=float).ravel()
    K = np.asarray(K, dtype=float)
    n = len(y)

    if K.shape != (n, n):
        raise ValueError(f"K must be ({n},{n}), got {K.shape}.")
    if lambda_val <= 0:
        raise ValueError("lambda_val must be positive.")

    alpha = np.linalg.solve(K + lambda_val * np.eye(n), y)
    y_hat = K @ alpha

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    corr = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0

    return GenomicsResult(
        name="RKHS",
        statistic=corr,
        n=n,
        extra={
            "r_squared": r_sq,
            "fitted": y_hat.tolist(),
            "alpha": alpha.tolist(),
        },
    )


def cheatsheet() -> str:
    return "rkhs(y, K) -> RKHS regression for genomic prediction."
