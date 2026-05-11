# morie.fn — function file (hadesllm/morie)
"""Gene-by-environment interaction test."""

__all__ = ["gxenv"]

import numpy as np
from scipy.stats import f as _f_dist

from ._containers import GenomicsResult


def gxenv(y: np.ndarray, genotypes: np.ndarray, environment: np.ndarray, cdf=None, *, covariates: np.ndarray | None = None) -> GenomicsResult:
    """Test for gene-by-environment (GxE) interaction.

    Fits full model y = b0 + b_g*g + b_e*env + b_ge*g*env + covs + e
    and tests H0: b_ge = 0 via F-test.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    genotypes : array, shape (n,)
        Marker genotypes (0/1/2).
    environment : array, shape (n,)
        Environmental exposure (continuous or binary).
    covariates : array, shape (n, q), optional
        Additional covariates.

    Returns
    -------
    GenomicsResult
        statistic = F-statistic for GxE interaction,
        p_value = p-value for interaction term.

    References
    ----------
    Thomas, D. (2010). Gene-environment-wide association studies:
        emerging approaches. Nature Reviews Genetics, 11(4), 259-272.
    """
    y = np.asarray(y, dtype=float).ravel()
    g = np.asarray(genotypes, dtype=float).ravel()
    env = np.asarray(environment, dtype=float).ravel()
    n = len(y)

    if len(g) != n or len(env) != n:
        raise ValueError("All arrays must have the same length.")

    base = [np.ones(n), g, env]
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        base.append(cov)

    X_reduced = np.column_stack(base)
    X_full = np.column_stack(base + [g * env])

    def _fit(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        return float(np.sum(resid ** 2)), beta

    rss_r, _ = _fit(X_reduced)
    rss_f, beta_full = _fit(X_full)

    df1 = 1
    df2 = max(n - X_full.shape[1], 1)

    if rss_f > 0 and rss_r >= rss_f:
        f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    else:
        f_stat = 0.0

    p_value = float(1.0 - _f_dist.cdf(f_stat, df1, df2))

    b_ge = float(beta_full[-1])

    return GenomicsResult(
        name="GxE_interaction",
        statistic=float(f_stat),
        p_value=p_value,
        n=n,
        extra={
            "beta_interaction": b_ge,
            "rss_full": rss_f,
            "rss_reduced": rss_r,
            "df1": df1,
            "df2": df2,
        },
    )


def cheatsheet() -> str:
    return "gxenv(y, geno, env) -> Gene-by-environment interaction test."
