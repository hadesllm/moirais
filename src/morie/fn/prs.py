# morie.fn — function file (hadesllm/morie)
"""Polygenic risk score calculation."""

import numpy as np

from ._containers import DescriptiveResult


def polygenic_risk_score(
    genotypes: np.ndarray, effect_sizes: np.ndarray, p_values: np.ndarray | None = None, p_threshold: float = 1.0
) -> DescriptiveResult:
    """
    Calculate polygenic risk scores (PRS) by summing effect-weighted genotypes.

    .. math::

        PRS_i = \\sum_{j \\in S} \\beta_j \\cdot G_{ij}

    where S is the set of SNPs passing the p-value threshold.

    :param genotypes: (n_samples, n_snps) genotype matrix coded 0/1/2.
    :param effect_sizes: (n_snps,) effect sizes (betas) from GWAS.
    :param p_values: (n_snps,) p-values for SNP filtering (optional).
    :param p_threshold: Include SNPs with p < threshold (default 1.0 = all).
    :return: DescriptiveResult with mean PRS as value.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Choi SW, Mak TSH, O'Reilly PF (2020). Tutorial: a guide to
    performing polygenic risk score analyses.
    Nature Protocols, 15(9), 2759-2772.
    """
    G = np.asarray(genotypes, dtype=np.float64)
    beta = np.asarray(effect_sizes, dtype=np.float64).ravel()
    if G.ndim == 1:
        G = G.reshape(1, -1)
    if G.shape[1] != len(beta):
        raise ValueError("Number of SNPs must match between genotypes and effect_sizes.")
    if p_values is not None:
        pv = np.asarray(p_values, dtype=np.float64).ravel()
        mask = pv < p_threshold
    else:
        mask = np.ones(len(beta), dtype=bool)
    prs = G[:, mask] @ beta[mask]
    return DescriptiveResult(
        name="polygenic_risk_score",
        value=float(prs.mean()),
        extra={
            "scores": prs,
            "n_samples": G.shape[0],
            "n_snps_used": int(mask.sum()),
            "n_snps_total": len(beta),
            "mean": float(prs.mean()),
            "sd": float(prs.std()),
        },
    )


prs = polygenic_risk_score


def cheatsheet() -> str:
    return "polygenic_risk_score({}) -> Polygenic risk score calculation."
