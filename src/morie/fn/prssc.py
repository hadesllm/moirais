# morie.fn — function file (hadesllm/morie)
"""Polygenic risk score calculation."""

__all__ = ["prssc"]

import numpy as np

from ._containers import GenomicsResult


def prssc(
    genotypes: np.ndarray,
    effect_sizes: np.ndarray,
    *,
    p_values: np.ndarray | None = None,
    p_threshold: float = 1.0,
    allele_freqs: np.ndarray | None = None,
    standardize: bool = False,
) -> GenomicsResult:
    """Compute polygenic risk scores (PRS).

    PRS_i = sum_j beta_j * g_ij

    where beta_j are GWAS effect sizes and g_ij are genotype
    dosages for individual i at SNP j.

    Parameters
    ----------
    genotypes : array, shape (n, p)
        Marker genotype matrix (0/1/2 or dosage).
    effect_sizes : array, shape (p,)
        GWAS effect sizes (log-OR or beta).
    p_values : array, shape (p,), optional
        GWAS p-values for thresholding.
    p_threshold : float
        Include only SNPs with p < p_threshold.
    allele_freqs : array, shape (p,), optional
        Allele frequencies for standardization.
    standardize : bool
        If True, standardize genotypes before scoring.

    Returns
    -------
    GenomicsResult
        statistic = mean PRS, extra has 'scores' (per-individual),
        'n_snps_used'.

    References
    ----------
    Purcell, S. M., et al. (2009). Common polygenic variation
        contributes to risk of schizophrenia and bipolar disorder.
        Nature, 460(7256), 748-752.
    Choi, S. W., Mak, T. S.-H., & O'Reilly, P. F. (2020). Tutorial:
        a guide to performing polygenic risk score analyses. Nature
        Protocols, 15(9), 2759-2772.
    """
    Z = np.asarray(genotypes, dtype=float)
    beta = np.asarray(effect_sizes, dtype=float).ravel()

    if Z.ndim != 2:
        raise ValueError("genotypes must be 2-D (n x p).")
    n, p = Z.shape
    if len(beta) != p:
        raise ValueError("effect_sizes length must match number of markers.")

    mask = np.ones(p, dtype=bool)
    if p_values is not None:
        pv = np.asarray(p_values, dtype=float).ravel()
        if len(pv) != p:
            raise ValueError("p_values length must match number of markers.")
        mask = pv < p_threshold

    n_used = int(np.sum(mask))

    if standardize and n_used > 0:
        if allele_freqs is not None:
            pf = np.asarray(allele_freqs, dtype=float).ravel()
        else:
            pf = np.mean(Z[:, mask], axis=0) / 2.0
        pf = np.clip(pf, 1e-6, 1 - 1e-6)
        sd = np.sqrt(2.0 * pf * (1.0 - pf))
        Z_use = (Z[:, mask] - 2.0 * pf) / sd
    else:
        Z_use = Z[:, mask]

    beta_use = beta[mask]
    scores = Z_use @ beta_use

    return GenomicsResult(
        name="PRS",
        statistic=float(np.mean(scores)),
        n=n,
        extra={
            "scores": scores.tolist(),
            "n_snps_used": n_used,
            "n_snps_total": p,
            "p_threshold": p_threshold,
            "sd": float(np.std(scores)),
        },
    )


def cheatsheet() -> str:
    return "prssc(geno, betas) -> Polygenic risk score calculation."
