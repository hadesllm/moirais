# moirais.fn — function file (hadesllm/moirais)
"""LD score regression."""

__all__ = ["ldscr"]

import numpy as np

from ._containers import GenomicsResult


def ldscr(
    chi2_stats: np.ndarray,
    ld_scores: np.ndarray,
    *,
    n_gwas: int = 1000,
    intercept_fixed: float | None = None,
) -> GenomicsResult:
    """LD score regression for heritability and confounding.

    Regresses GWAS chi-squared statistics on LD scores:

        E[chi2_j] = 1 + N * h2 * l_j / M + a

    where l_j is the LD score and a captures confounding.

    Parameters
    ----------
    chi2_stats : array, shape (p,)
        GWAS chi-squared statistics per SNP.
    ld_scores : array, shape (p,)
        LD scores per SNP.
    n_gwas : int
        GWAS sample size.
    intercept_fixed : float, optional
        If set, fix the intercept to this value (e.g. 1.0).

    Returns
    -------
    GenomicsResult
        statistic = h2_snp estimate,
        extra has 'intercept', 'slope', 'se_h2', 'mean_chi2',
        'lambda_gc'.

    References
    ----------
    Bulik-Sullivan, B. K., et al. (2015). LD Score regression
        distinguishes confounding from polygenicity in genome-wide
        association studies. Nature Genetics, 47(3), 291-295.
    """
    chi2 = np.asarray(chi2_stats, dtype=float).ravel()
    ld = np.asarray(ld_scores, dtype=float).ravel()
    p = len(chi2)

    if len(ld) != p:
        raise ValueError("chi2_stats and ld_scores must have same length.")
    if p < 2:
        raise ValueError("Need at least 2 SNPs.")
    if n_gwas < 1:
        raise ValueError("n_gwas must be positive.")

    if intercept_fixed is not None:
        y = chi2 - intercept_fixed
        slope = float(np.sum(ld * y) / np.sum(ld ** 2))
        intercept = intercept_fixed
        resid = y - slope * ld
    else:
        X = np.column_stack([np.ones(p), ld])
        beta = np.linalg.lstsq(X, chi2, rcond=None)[0]
        intercept = float(beta[0])
        slope = float(beta[1])
        resid = chi2 - X @ beta

    M = p
    h2 = slope * M / n_gwas

    mse = np.sum(resid ** 2) / max(p - 2, 1)
    ld_var = np.var(ld)
    se_slope = np.sqrt(mse / (p * ld_var)) if ld_var > 0 else float("nan")
    se_h2 = se_slope * M / n_gwas

    mean_chi2 = float(np.mean(chi2))
    lambda_gc = float(np.median(chi2)) / 0.4549364

    return GenomicsResult(
        name="LDSC",
        statistic=float(h2),
        n=p,
        extra={
            "intercept": intercept,
            "slope": slope,
            "se_h2": float(se_h2),
            "mean_chi2": mean_chi2,
            "lambda_gc": lambda_gc,
            "n_gwas": n_gwas,
            "n_snps": p,
        },
    )


def cheatsheet() -> str:
    return "ldscr(chi2, ld_scores) -> LD score regression for h2."
