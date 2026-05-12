# morie.fn -- function file (hadesllm/morie)
"""Single-SNP GWAS association test."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import GenomicsResult


def gwas_single_snp(
    genotypes: np.ndarray, phenotype: np.ndarray, covariates: np.ndarray | None = None
) -> GenomicsResult:
    """
    Single-SNP association test for GWAS.

    Fits a linear regression of phenotype on genotype (additive model)
    with optional covariates. Returns the genotype effect size, SE,
    and p-value from a t-test.

    :param genotypes: (n,) array coded 0/1/2 for one SNP.
    :param phenotype: (n,) continuous outcome.
    :param covariates: (n, k) covariate matrix or None.
    :return: GenomicsResult with t-statistic and p-value.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Marchini J, Howie B (2010). Genotype imputation for genome-wide
    association studies. Nature Reviews Genetics, 11(7), 499-511.
    """
    g = np.asarray(genotypes, dtype=np.float64).ravel()
    y = np.asarray(phenotype, dtype=np.float64).ravel()
    n = len(g)
    if len(y) != n:
        raise ValueError("genotypes and phenotype must have same length.")
    if covariates is not None:
        C = np.asarray(covariates, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        X = np.column_stack([np.ones(n), g, C])
    else:
        X = np.column_stack([np.ones(n), g])
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    df = n - X.shape[1]
    mse = rss / df if df > 0 else 1e-10
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_beta = np.sqrt(np.diag(XtX_inv) * mse)
    t_stat = beta[1] / se_beta[1] if se_beta[1] > 0 else 0.0
    p_val = float(2 * sp_stats.t.sf(abs(t_stat), df))
    return GenomicsResult(
        name="gwas_single_snp",
        statistic=float(t_stat),
        p_value=p_val,
        n=n,
        extra={"beta": float(beta[1]), "se": float(se_beta[1]), "df": df, "mse": mse},
    )


gwas1 = gwas_single_snp


def cheatsheet() -> str:
    return "gwas_single_snp({}) -> Single-SNP GWAS association test."
