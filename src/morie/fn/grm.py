# morie.fn -- function file (hadesllm/morie)
"""Genomic relationship matrix (GRM)."""

import numpy as np

from ._containers import DescriptiveResult


def genetic_relatedness(genotype_matrix: np.ndarray) -> DescriptiveResult:
    r"""
    Compute the genomic relationship matrix (GRM).

    Uses the method of VanRaden (2008):

    .. math::

        G = \\frac{Z Z^T}{2 \\sum p_j(1-p_j)}

    where Z is centered by allele frequencies.

    :param genotype_matrix: (n_samples, n_snps) array coded 0/1/2.
    :return: DescriptiveResult with mean diagonal as value, GRM in extra.
    :raises ValueError: If matrix is not 2D.

    References
    ----------
    VanRaden PM (2008). Efficient methods to compute genomic predictions.
    Journal of Dairy Science, 91(11), 4414-4423.
    """
    G = np.asarray(genotype_matrix, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("genotype_matrix must be 2D.")
    n, p = G.shape
    freq = G.mean(axis=0) / 2.0
    freq = np.clip(freq, 1e-10, 1 - 1e-10)
    Z = G - 2 * freq
    denom = 2.0 * np.sum(freq * (1 - freq))
    grm = Z @ Z.T / denom if denom > 0 else np.eye(n)
    mean_diag = float(np.mean(np.diag(grm)))
    return DescriptiveResult(
        name="genetic_relatedness",
        value=mean_diag,
        extra={"grm": grm, "n_samples": n, "n_snps": p, "allele_freq": freq},
    )


grm = genetic_relatedness


def cheatsheet() -> str:
    return "genetic_relatedness({}) -> Genomic relationship matrix (GRM)."
