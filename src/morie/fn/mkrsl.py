# morie.fn — function file (hadesllm/morie)
"""Marker selection (variable importance for genomic prediction)."""

__all__ = ["mkrsl"]

import numpy as np

from ._containers import GenomicsResult


def mkrsl(
    y: np.ndarray,
    Z: np.ndarray,
    *,
    method: str = "marginal",
    n_select: int | None = None,
    alpha: float = 0.05,
) -> GenomicsResult:
    """Select informative markers for genomic prediction.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2).
    method : {'marginal', 'ridge_importance', 'correlation'}
        Selection method:
        - 'marginal': single-SNP regression p-values
        - 'ridge_importance': squared ridge coefficients
        - 'correlation': absolute Pearson correlation
    n_select : int, optional
        Number of markers to select.  If None, selects those
        passing alpha threshold (marginal) or top 10% (others).
    alpha : float
        Significance threshold for marginal method.

    Returns
    -------
    GenomicsResult
        statistic = number of selected markers,
        extra has 'selected_indices', 'scores' (importance per marker).

    References
    ----------
    Meuwissen, T. H. E., et al. (2001). Prediction of total genetic
        value using genome-wide dense marker maps. Genetics, 157(4),
        1819-1829.
    """
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape

    if n != len(y):
        raise ValueError("Z rows must match length of y.")

    scores = np.zeros(p)

    if method == "marginal":
        from scipy.stats import t as _t_dist
        y_c = y - np.mean(y)
        for j in range(p):
            z_c = Z[:, j] - np.mean(Z[:, j])
            ss_z = np.sum(z_c ** 2)
            if ss_z < 1e-12:
                scores[j] = 0.0
                continue
            beta = np.sum(z_c * y_c) / ss_z
            resid = y_c - beta * z_c
            se = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1) / ss_z)
            if se > 1e-12:
                t_val = abs(beta / se)
                scores[j] = float(2 * _t_dist.sf(t_val, n - 2))
            else:
                scores[j] = 1.0

        if n_select is None:
            selected = np.where(scores < alpha)[0]
        else:
            selected = np.argsort(scores)[:n_select]
        importance = 1.0 - scores

    elif method == "ridge_importance":
        lam = 1.0
        ZtZ = Z.T @ Z + lam * np.eye(p)
        beta = np.linalg.solve(ZtZ, Z.T @ y)
        scores = beta ** 2
        importance = scores

        if n_select is None:
            n_select = max(p // 10, 1)
        selected = np.argsort(scores)[::-1][:n_select]

    elif method == "correlation":
        for j in range(p):
            if np.std(Z[:, j]) > 1e-12:
                scores[j] = abs(np.corrcoef(Z[:, j], y)[0, 1])
            else:
                scores[j] = 0.0
        importance = scores

        if n_select is None:
            n_select = max(p // 10, 1)
        selected = np.argsort(scores)[::-1][:n_select]

    else:
        raise ValueError(f"Unknown method: {method}")

    return GenomicsResult(
        name="MarkerSelection",
        statistic=float(len(selected)),
        n=n,
        extra={
            "selected_indices": selected.tolist(),
            "scores": importance.tolist(),
            "method": method,
            "n_markers": p,
        },
    )


def cheatsheet() -> str:
    return "mkrsl(y, Z) -> Marker selection for genomic prediction."
