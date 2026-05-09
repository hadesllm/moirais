"""Sex-stratified meta-regression."""

__all__ = ["sxmrg"]

import numpy as np
from scipy.stats import norm as _norm

from ._containers import GenomicsResult


def sxmrg(
    betas_male: np.ndarray,
    ses_male: np.ndarray,
    betas_female: np.ndarray,
    ses_female: np.ndarray,
) -> GenomicsResult:
    """Sex-stratified meta-regression to test for sex-differential effects.

    Combines sex-specific GWAS summary statistics to test whether
    genetic effects differ between males and females using a
    two-sample z-test on the difference.

    Parameters
    ----------
    betas_male : array, shape (p,)
        Effect estimates from male-only GWAS.
    ses_male : array, shape (p,)
        Standard errors from male-only GWAS.
    betas_female : array, shape (p,)
        Effect estimates from female-only GWAS.
    ses_female : array, shape (p,)
        Standard errors from female-only GWAS.

    Returns
    -------
    GenomicsResult
        statistic = number of significant sex-differential SNPs
        (at alpha=0.05 after Bonferroni),
        extra has 'z_diff', 'p_diff', 'beta_diff' arrays.

    References
    ----------
    Randall, J. C., et al. (2013). Sex-stratified genome-wide
        association studies including 270,000 individuals show
        sexual dimorphism in genetic loci for anthropometric
        traits. PLoS Genetics, 9(6), e1003500.
    """
    bm = np.asarray(betas_male, dtype=float).ravel()
    sm = np.asarray(ses_male, dtype=float).ravel()
    bf = np.asarray(betas_female, dtype=float).ravel()
    sf = np.asarray(ses_female, dtype=float).ravel()

    p = len(bm)
    if len(sm) != p or len(bf) != p or len(sf) != p:
        raise ValueError("All arrays must have the same length.")

    if np.any(sm <= 0) or np.any(sf <= 0):
        raise ValueError("Standard errors must be positive.")

    beta_diff = bm - bf
    se_diff = np.sqrt(sm ** 2 + sf ** 2)
    z_diff = beta_diff / se_diff
    p_diff = 2.0 * _norm.sf(np.abs(z_diff))

    threshold = 0.05 / p if p > 0 else 0.05
    n_sig = int(np.sum(p_diff < threshold))

    wm = 1.0 / sm ** 2
    wf = 1.0 / sf ** 2
    beta_combined = (wm * bm + wf * bf) / (wm + wf)
    se_combined = 1.0 / np.sqrt(wm + wf)

    return GenomicsResult(
        name="SexStratifiedMeta",
        statistic=float(n_sig),
        n=p,
        extra={
            "z_diff": z_diff.tolist(),
            "p_diff": p_diff.tolist(),
            "beta_diff": beta_diff.tolist(),
            "se_diff": se_diff.tolist(),
            "beta_combined": beta_combined.tolist(),
            "se_combined": se_combined.tolist(),
            "bonferroni_threshold": threshold,
        },
    )


def cheatsheet() -> str:
    return "sxmrg(bm, sm, bf, sf) -> Sex-stratified meta-regression."
