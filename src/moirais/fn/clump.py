# moirais.fn — function file (hadesllm/moirais)
"""LD-based clumping of significant SNPs."""

import numpy as np

from ._containers import DescriptiveResult


def ld_clumping(
    p_values: np.ndarray, ld_matrix: np.ndarray, threshold: float = 5e-8, r2_threshold: float = 0.1
) -> DescriptiveResult:
    """
    LD-based clumping to select independent significant SNPs.

    Iteratively selects the most significant SNP and removes all SNPs
    in LD (r^2 > r2_threshold) with it.

    :param p_values: Array of p-values for each SNP.
    :param ld_matrix: (n_snps, n_snps) LD (r^2) matrix.
    :param threshold: Significance threshold for index SNPs.
    :param r2_threshold: LD r^2 threshold for clumping.
    :return: DescriptiveResult with index SNP indices.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Purcell S et al. (2007). PLINK: a tool set for whole-genome
    association and population-based linkage analyses.
    AJHG, 81(3), 559-575.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    ld = np.asarray(ld_matrix, dtype=np.float64)
    n = len(pv)
    if ld.shape != (n, n):
        raise ValueError("ld_matrix must be (n_snps, n_snps).")
    sig_mask = pv < threshold
    candidates = set(np.where(sig_mask)[0])
    index_snps = []
    while candidates:
        best = min(candidates, key=lambda i: pv[i])
        index_snps.append(int(best))
        candidates.discard(best)
        to_remove = {j for j in candidates if ld[best, j] > r2_threshold}
        candidates -= to_remove
    return DescriptiveResult(
        name="ld_clumping",
        value=len(index_snps),
        extra={
            "index_snps": np.array(index_snps),
            "n_total": n,
            "n_significant": int(sig_mask.sum()),
            "threshold": threshold,
            "r2_threshold": r2_threshold,
        },
    )


clump = ld_clumping


def cheatsheet() -> str:
    return "ld_clumping({}) -> LD-based clumping of significant SNPs."
