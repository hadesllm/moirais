# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Simple admixture estimation via NMF on genotype matrix."""

import numpy as np

from ._containers import DescriptiveResult


def admixture_proportions(
    genotype_matrix: np.ndarray, K: int = 2, n_iter: int = 200, seed: int = 42
) -> DescriptiveResult:
    """
    Estimate admixture proportions using non-negative matrix factorization.

    Approximates STRUCTURE/ADMIXTURE by decomposing the genotype matrix
    G ~ Q * F where Q (n x K) gives individual admixture proportions
    and F (K x p) gives population-specific allele frequencies.

    :param genotype_matrix: (n_samples, n_snps) array coded 0/1/2.
    :param K: Number of ancestral populations.
    :param n_iter: Number of multiplicative update iterations.
    :param seed: Random seed.
    :return: DescriptiveResult with Q matrix in extra.
    :raises ValueError: If K < 2.

    References
    ----------
    Alexander DH, Novembre J, Lange K (2009). Fast model-based
    estimation of ancestry in unrelated individuals.
    Genome Research, 19(9), 1655-1664.
    """
    if K < 2:
        raise ValueError("K must be >= 2.")
    G = np.asarray(genotype_matrix, dtype=np.float64)
    n, p = G.shape
    rng = np.random.default_rng(seed)
    Q = rng.random((n, K)) + 0.1
    F = rng.random((K, p)) + 0.1
    G_pos = np.clip(G, 0, 2)
    for _ in range(n_iter):
        F *= (Q.T @ G_pos) / (Q.T @ Q @ F + 1e-10)
        Q *= (G_pos @ F.T) / (Q @ F @ F.T + 1e-10)
    Q_norm = Q / Q.sum(axis=1, keepdims=True)
    residual = float(np.linalg.norm(G_pos - Q @ F, "fro"))
    return DescriptiveResult(
        name="admixture_proportions",
        value=float(K),
        extra={"Q": Q_norm, "F": F, "K": K, "residual": residual, "n_samples": n, "n_snps": p},
    )


admix = admixture_proportions


def cheatsheet() -> str:
    return "admixture_proportions({}) -> Simple admixture estimation via NMF on genotype matrix."
