"""Sex interaction test (GxSex) for genetic association."""

__all__ = ["sxint"]

import numpy as np
from scipy.stats import chi2 as _chi2

from ._containers import GenomicsResult


def sxint(y: np.ndarray, genotypes: np.ndarray, sex: np.ndarray, cdf=None, *, covariates: np.ndarray | None = None) -> GenomicsResult:
    """Test for gene-by-sex interaction (GxSex).

    Compares a full model (with GxSex interaction) to a reduced
    model (no interaction) using a likelihood ratio test.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    genotypes : array, shape (n,)
        Marker genotypes (0/1/2).
    sex : array, shape (n,)
        Sex indicator (0/1).
    covariates : array, shape (n, q), optional
        Additional covariates.

    Returns
    -------
    GenomicsResult
        statistic = chi-squared test statistic (1 df),
        p_value = p-value for GxSex interaction.

    References
    ----------
    Kraft, P., et al. (2007). Exploiting gene-environment
        interaction to detect genetic associations. Human
        Heredity, 63(2), 111-119.
    """
    y = np.asarray(y, dtype=float).ravel()
    g = np.asarray(genotypes, dtype=float).ravel()
    s = np.asarray(sex, dtype=float).ravel()
    n = len(y)

    if len(g) != n or len(s) != n:
        raise ValueError("All arrays must have same length.")

    base_cols = [np.ones(n), g, s]
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        base_cols.append(cov)

    X_reduced = np.column_stack(base_cols)
    X_full = np.column_stack(base_cols + [g * s])

    def _rss(X, y_vec):
        try:
            beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.sum(y_vec ** 2)
        return float(np.sum((y_vec - X @ beta) ** 2))

    rss_reduced = _rss(X_reduced, y)
    rss_full = _rss(X_full, y)

    df_diff = 1
    df_full = max(n - X_full.shape[1], 1)

    if rss_full > 0 and rss_reduced > rss_full:
        f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_full)
        chi2_stat = f_stat * df_diff
    else:
        chi2_stat = 0.0

    p_value = float(1.0 - _chi2.cdf(chi2_stat, df_diff))

    return GenomicsResult(
        name="GxSex_interaction",
        statistic=float(chi2_stat),
        p_value=p_value,
        n=n,
        extra={
            "rss_full": rss_full,
            "rss_reduced": rss_reduced,
            "df": df_diff,
        },
    )


def cheatsheet() -> str:
    return "sxint(y, geno, sex) -> Gene-by-sex interaction test."
