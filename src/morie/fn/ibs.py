# morie.fn -- function file (hadesllm/morie)
"""Identity-by-state similarity matrix."""

import numpy as np

from ._containers import DescriptiveResult


def identity_by_state(genotype_matrix: np.ndarray) -> DescriptiveResult:
    """
    Compute pairwise identity-by-state (IBS) similarity matrix.

    For each pair of individuals, IBS is the proportion of loci
    sharing the same genotype. IBS(i,j) in [0,1].

    :param genotype_matrix: (n_samples, n_snps) array coded 0/1/2.
    :return: DescriptiveResult with mean IBS as value, matrix in extra.
    :raises ValueError: If matrix is not 2D.

    References
    ----------
    Purcell S et al. (2007). PLINK: a tool set for whole-genome
    association and population-based linkage analyses.
    AJHG, 81(3), 559-575.
    """
    G = np.asarray(genotype_matrix, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("genotype_matrix must be 2D.")
    n, p = G.shape
    ibs_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            match = np.sum(G[i] == G[j])
            similarity = match / p
            ibs_mat[i, j] = similarity
            ibs_mat[j, i] = similarity
    mean_ibs = float((ibs_mat.sum() - n) / (n * (n - 1))) if n > 1 else 1.0
    return DescriptiveResult(
        name="identity_by_state",
        value=mean_ibs,
        extra={"ibs_matrix": ibs_mat, "n_samples": n, "n_snps": p},
    )


ibs = identity_by_state


def cheatsheet() -> str:
    return "identity_by_state({}) -> Identity-by-state similarity matrix."
