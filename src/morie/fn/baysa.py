# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BayesA — t-distributed marker effects for genomic prediction."""

__all__ = ["baysa"]

import numpy as np

from ._containers import GenomicsResult


def baysa(
    y: np.ndarray,
    Z: np.ndarray,
    *,
    n_iter: int = 1000,
    burn_in: int = 200,
    df_prior: float = 4.0,
    scale_prior: float | None = None,
    seed: int = 42,
) -> GenomicsResult:
    """BayesA genomic prediction via Gibbs sampling.

    Each marker effect has its own variance drawn from a scaled
    inverse-chi-squared prior, producing t-distributed marginal effects.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector (centered recommended).
    Z : array, shape (n, p)
        Marker genotype matrix (coded 0/1/2).
    n_iter : int
        Total MCMC iterations.
    burn_in : int
        Burn-in iterations to discard.
    df_prior : float
        Degrees of freedom for the scaled inverse-chi-squared prior
        on each marker variance.
    scale_prior : float, optional
        Scale parameter for the marker variance prior.  If None,
        estimated as var(y) / p * (df_prior - 2) / df_prior.
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = correlation(y, y_hat),
        extra has 'effects' (posterior mean marker effects)
        and 'var_e' (residual variance estimate).

    References
    ----------
    Meuwissen, T. H. E., Hayes, B. J., & Goddard, M. E. (2001).
        Prediction of total genetic value using genome-wide dense
        marker maps. Genetics, 157(4), 1819-1829.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape

    if n != len(y):
        raise ValueError("Z rows must match length of y.")

    if scale_prior is None:
        scale_prior = np.var(y) / p * (df_prior - 2.0) / df_prior
        scale_prior = max(scale_prior, 1e-8)

    ztz = np.sum(Z ** 2, axis=0)

    g = np.zeros(p)
    var_e = np.var(y) * 0.5
    var_m = np.full(p, scale_prior)
    e = y - Z @ g

    effects_accum = np.zeros(p)
    var_e_accum = 0.0
    n_saved = 0

    for it in range(n_iter):
        for j in range(p):
            e += Z[:, j] * g[j]
            rhs = Z[:, j] @ e
            lhs = ztz[j] + var_e / max(var_m[j], 1e-12)
            mean_j = rhs / lhs
            var_j = var_e / lhs
            g[j] = rng.normal(mean_j, np.sqrt(max(var_j, 1e-12)))
            e -= Z[:, j] * g[j]

        for j in range(p):
            shape = (df_prior + 1.0) / 2.0
            scale_post = (df_prior * scale_prior + g[j] ** 2) / 2.0
            var_m[j] = scale_post / rng.gamma(shape)

        shape_e = (n - 2.0) / 2.0
        scale_e = np.sum(e ** 2) / 2.0
        if shape_e > 0:
            var_e = scale_e / rng.gamma(shape_e)

        if it >= burn_in:
            effects_accum += g
            var_e_accum += var_e
            n_saved += 1

    effects = effects_accum / max(n_saved, 1)
    var_e_est = var_e_accum / max(n_saved, 1)
    y_hat = Z @ effects
    corr = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0

    return GenomicsResult(
        name="BayesA",
        statistic=corr,
        n=n,
        extra={
            "effects": effects.tolist(),
            "var_e": float(var_e_est),
            "n_iter": n_iter,
            "burn_in": burn_in,
        },
    )


def cheatsheet() -> str:
    return "baysa(y, Z) -> BayesA genomic prediction (t-distributed marker effects)."
