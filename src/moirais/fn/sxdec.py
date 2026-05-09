"""Sex-specific effect decomposition."""

__all__ = ["sxdec"]

import numpy as np

from ._containers import GenomicsResult


def sxdec(
    y: np.ndarray,
    genotypes: np.ndarray,
    sex: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
) -> GenomicsResult:
    """Decompose genetic effects into sex-specific components.

    Fits a model: y = b0 + b_g * g + b_s * sex + b_gs * g * sex + e

    and decomposes the genetic effect into:
    - Male-specific effect: b_g + b_gs (when sex=1)
    - Female-specific effect: b_g (when sex=0)
    - Sex-differential effect: b_gs

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    genotypes : array, shape (n,) or (n, p)
        Marker genotypes (0/1/2).  If 2-D, effects are computed
        for each marker separately.
    sex : array, shape (n,)
        Sex indicator (0 = female/reference, 1 = male/alternate).
    covariates : array, shape (n, q), optional
        Additional covariates.

    Returns
    -------
    GenomicsResult
        statistic = sex-differential effect (interaction beta),
        extra has 'beta_male', 'beta_female', 'beta_interaction',
        'se_interaction', 'p_interaction'.

    References
    ----------
    Rawlik, K., Canela-Xandri, O., & Tenesa, A. (2016). Evidence
        for sex-specific genetic architectures across a spectrum
        of human complex traits. Genome Biology, 17, 166.
    """
    y = np.asarray(y, dtype=float).ravel()
    genotypes = np.asarray(genotypes, dtype=float)
    sex = np.asarray(sex, dtype=float).ravel()
    n = len(y)

    if genotypes.ndim == 1:
        genotypes = genotypes.reshape(-1, 1)

    if len(sex) != n or genotypes.shape[0] != n:
        raise ValueError("All arrays must have the same number of observations.")

    results = []
    for j in range(genotypes.shape[1]):
        g = genotypes[:, j]
        gs = g * sex

        if covariates is not None:
            cov = np.asarray(covariates, dtype=float)
            X = np.column_stack([np.ones(n), g, sex, gs, cov])
        else:
            X = np.column_stack([np.ones(n), g, sex, gs])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(X.shape[1])

        y_hat = X @ beta
        resid = y - y_hat
        df = max(n - X.shape[1], 1)
        mse = np.sum(resid ** 2) / df

        try:
            cov_beta = mse * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            cov_beta = np.eye(X.shape[1]) * mse

        b_g = beta[1]
        b_gs = beta[3]
        se_gs = np.sqrt(max(cov_beta[3, 3], 0))

        from scipy.stats import t as t_dist
        if se_gs > 1e-10:
            t_val = b_gs / se_gs
            p_val = float(2 * t_dist.sf(abs(t_val), df))
        else:
            p_val = 1.0

        results.append({
            "beta_female": float(b_g),
            "beta_male": float(b_g + b_gs),
            "beta_interaction": float(b_gs),
            "se_interaction": float(se_gs),
            "p_interaction": p_val,
        })

    if len(results) == 1:
        r = results[0]
        return GenomicsResult(
            name="SexDecomposition",
            statistic=r["beta_interaction"],
            p_value=r["p_interaction"],
            n=n,
            extra=r,
        )

    return GenomicsResult(
        name="SexDecomposition",
        statistic=float(np.mean([r["beta_interaction"] for r in results])),
        n=n,
        extra={"markers": results},
    )


def cheatsheet() -> str:
    return "sxdec(y, geno, sex) -> Sex-specific genetic effect decomposition."
