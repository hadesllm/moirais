# morie.fn — function file (hadesllm/morie)
"""Population structure via PCA on genotype data."""

__all__ = ["popst"]

import numpy as np

from ._containers import GenomicsResult


def popst(
    Z: np.ndarray,
    *,
    n_components: int = 2,
) -> GenomicsResult:
    """Estimate population structure using PCA on genotypes.

    Performs eigendecomposition of the centered-and-scaled genotype
    matrix to extract principal components that capture population
    stratification.

    Parameters
    ----------
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2).
    n_components : int
        Number of principal components to return.

    Returns
    -------
    GenomicsResult
        statistic = proportion of variance explained by first PC,
        extra has 'scores' (n x n_components), 'var_explained'
        (proportion for each PC), 'eigenvalues'.

    References
    ----------
    Price, A. L., et al. (2006). Principal components analysis
        corrects for stratification in genome-wide association
        studies. Nature Genetics, 38(8), 904-909.
    Patterson, N., Price, A. L., & Reich, D. (2006). Population
        structure and eigenanalysis. PLoS Genetics, 2(12), e190.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D.")
    n, p = Z.shape
    if n < 2:
        raise ValueError("Need at least 2 individuals.")

    n_components = min(n_components, min(n, p))

    pf = np.mean(Z, axis=0) / 2.0
    pf = np.clip(pf, 1e-6, 1.0 - 1e-6)
    denom = np.sqrt(2.0 * pf * (1.0 - pf))
    M = (Z - 2.0 * pf) / denom

    cov_mat = (M @ M.T) / p
    eigvals, eigvecs = np.linalg.eigh(cov_mat)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = np.sum(np.maximum(eigvals, 0))
    var_explained = eigvals[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
    scores = eigvecs[:, :n_components]

    return GenomicsResult(
        name="PopulationStructure_PCA",
        statistic=float(var_explained[0]) if len(var_explained) > 0 else 0.0,
        n=n,
        extra={
            "scores": scores.tolist(),
            "var_explained": var_explained.tolist(),
            "eigenvalues": eigvals[:n_components].tolist(),
            "n_markers": p,
        },
    )


def cheatsheet() -> str:
    return "popst(Z) -> Population structure estimation via PCA."
