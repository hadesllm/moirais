# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BayesB — spike-and-slab prior for genomic prediction."""

__all__ = ["baysb"]

import numpy as np

from ._containers import GenomicsResult


def baysb(
    y: np.ndarray,
    Z: np.ndarray,
    *,
    n_iter: int = 1000,
    burn_in: int = 200,
    pi: float = 0.95,
    df_prior: float = 4.0,
    scale_prior: float | None = None,
    seed: int = 42,
) -> GenomicsResult:
    """BayesB genomic prediction via Gibbs sampling.

    Uses a spike-and-slab prior: with probability pi the marker
    effect is exactly zero (spike), otherwise it has a scaled
    inverse-chi-squared variance (slab).

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    Z : array, shape (n, p)
        Marker genotype matrix (0/1/2).
    n_iter : int
        Total MCMC iterations.
    burn_in : int
        Burn-in to discard.
    pi : float
        Prior probability that a marker has zero effect.
    df_prior : float
        Degrees of freedom for slab variance prior.
    scale_prior : float, optional
        Scale for slab variance prior.
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = cor(y, y_hat), extra has 'effects',
        'inclusion_prob' (posterior inclusion for each marker).

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
        n_qtl = max(int(p * (1 - pi)), 1)
        scale_prior = np.var(y) / n_qtl * (df_prior - 2.0) / df_prior
        scale_prior = max(scale_prior, 1e-8)

    ztz = np.sum(Z ** 2, axis=0)

    g = np.zeros(p)
    var_e = np.var(y) * 0.5
    var_m = np.full(p, scale_prior)
    delta = np.zeros(p, dtype=int)
    e = y - Z @ g

    effects_accum = np.zeros(p)
    incl_accum = np.zeros(p)
    n_saved = 0

    for it in range(n_iter):
        for j in range(p):
            e += Z[:, j] * g[j]
            rhs = Z[:, j] @ e
            lhs_slab = ztz[j] + var_e / max(var_m[j], 1e-12)
            mean_slab = rhs / lhs_slab
            var_slab = var_e / lhs_slab

            log_bf_slab = (
                0.5 * np.log(var_e / (var_m[j] * lhs_slab + 1e-30))
                + 0.5 * mean_slab ** 2 / var_slab
            )
            log_odds = np.log(max(1 - pi, 1e-12)) - np.log(max(pi, 1e-12)) + log_bf_slab
            prob_in = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -500, 500)))

            if rng.uniform() < prob_in:
                delta[j] = 1
                g[j] = rng.normal(mean_slab, np.sqrt(max(var_slab, 1e-12)))
            else:
                delta[j] = 0
                g[j] = 0.0
            e -= Z[:, j] * g[j]

        for j in range(p):
            if delta[j] == 1:
                shape = (df_prior + 1.0) / 2.0
                scale_post = (df_prior * scale_prior + g[j] ** 2) / 2.0
                var_m[j] = scale_post / rng.gamma(shape)

        shape_e = (n - 2.0) / 2.0
        scale_e = np.sum(e ** 2) / 2.0
        if shape_e > 0:
            var_e = scale_e / rng.gamma(shape_e)

        if it >= burn_in:
            effects_accum += g
            incl_accum += delta
            n_saved += 1

    effects = effects_accum / max(n_saved, 1)
    incl_prob = incl_accum / max(n_saved, 1)
    y_hat = Z @ effects
    corr = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0

    return GenomicsResult(
        name="BayesB",
        statistic=corr,
        n=n,
        extra={
            "effects": effects.tolist(),
            "inclusion_prob": incl_prob.tolist(),
            "var_e": float(var_e),
            "pi": pi,
        },
    )


def cheatsheet() -> str:
    return "baysb(y, Z) -> BayesB genomic prediction (spike-and-slab prior)."
