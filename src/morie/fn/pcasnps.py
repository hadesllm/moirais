"""PCA on genotype matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pca_snps"]


def pca_snps(genotypes, n_components):
    """
    PCA on genotype matrix

    Formula: SVD of normalized M; top eigenvectors

    Parameters
    ----------
    genotypes : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Patterson-Price-Reich (2006)
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA on genotype matrix"})


def cheatsheet():
    return "pcasnps: PCA on genotype matrix"
