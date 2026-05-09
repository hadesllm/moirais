# moirais.fn — function file (hadesllm/moirais)
"""PCA on genotype matrix for population stratification."""

import numpy as np

from ._containers import DescriptiveResult


def pca_genotype(genotype_matrix: np.ndarray, n_components: int = 2) -> DescriptiveResult:
    """
    Principal component analysis on a genotype matrix (0/1/2 coded).

    Centers and standardises the genotype matrix by allele frequency,
    then computes PCA via SVD. Used for detecting population
    stratification in genetic association studies.

    :param genotype_matrix: (n_samples, n_snps) array with values 0, 1, 2.
    :param n_components: Number of PCs to retain (default 2).
    :return: DescriptiveResult with explained variance ratio as value.
    :raises ValueError: If matrix has wrong shape.

    References
    ----------
    Price AL et al. (2006). Principal components analysis corrects for
    stratification in genome-wide association studies.
    Nature Genetics, 38(8), 904-909.
    """
    G = np.asarray(genotype_matrix, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("genotype_matrix must be 2D.")
    n, p = G.shape
    freq = G.mean(axis=0) / 2.0
    freq = np.clip(freq, 1e-10, 1 - 1e-10)
    G_std = (G - 2 * freq) / np.sqrt(2 * freq * (1 - freq))
    U, s, Vt = np.linalg.svd(G_std, full_matrices=False)
    eig = s**2 / (n - 1)
    total_var = eig.sum()
    k = min(n_components, len(eig))
    scores = U[:, :k] * s[:k]
    var_ratio = eig[:k] / total_var
    return DescriptiveResult(
        name="pca_genotype",
        value=float(var_ratio.sum()),
        extra={
            "scores": scores,
            "eigenvalues": eig[:k],
            "variance_ratio": var_ratio,
            "n_components": k,
            "n_samples": n,
            "n_snps": p,
        },
    )


pca_g = pca_genotype


def cheatsheet() -> str:
    return "pca_genotype({}) -> PCA on genotype matrix for population stratification."
