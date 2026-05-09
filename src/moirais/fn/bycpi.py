# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BayesCpi — common variance mixture for genomic prediction."""

__all__ = ["bycpi"]

import numpy as np

from ._containers import GenomicsResult


def bycpi(
    y: np.ndarray,
    Z: np.ndarray,
    *,
    n_iter: int = 1000,
    burn_in: int = 200,
    pi_init: float = 0.95,
    df_prior: float = 4.0,
    scale_prior: float | None = None,
    seed: int = 42,
) -> GenomicsResult:
    """BayesCpi genomic prediction via Gibbs sampling.

    Like BayesB but all non-zero markers share a common variance,
    and pi is estimated from the data via a Beta(1,1) prior.

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
    pi_init : float
        Initial value for pi (proportion of zero-effect markers).
    df_prior : float
        Degrees of freedom for the common marker variance prior.
    scale_prior : float, optional
        Scale for marker variance prior.
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = cor(y, y_hat), extra has 'effects',
        'pi_est' (posterior mean of pi).

    References
    ----------
    Habier, D., Fernando, R. L., Kizilkaya, K., & Garrick, D. J.
        (2011). Extension of the Bayesian alphabet for genomic
        selection. BMC Bioinformatics, 12, 186.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape

    if n != len(y):
        raise ValueError("Z rows must match length of y.")

    if scale_prior is None:
        n_qtl = max(int(p * (1 - pi_init)), 1)
        scale_prior = np.var(y) / n_qtl * (df_prior - 2.0) / df_prior
        scale_prior = max(scale_prior, 1e-8)

    ztz = np.sum(Z ** 2, axis=0)
    g = np.zeros(p)
    var_e = np.var(y) * 0.5
    var_g = float(scale_prior)
    pi = pi_init
    delta = np.zeros(p, dtype=int)
    e = y - Z @ g

    effects_accum = np.zeros(p)
    pi_accum = 0.0
    n_saved = 0

    for it in range(n_iter):
        for j in range(p):
            e += Z[:, j] * g[j]
            rhs = Z[:, j] @ e
            lhs = ztz[j] + var_e / max(var_g, 1e-12)
            mean_j = rhs / lhs
            var_j = var_e / lhs

            log_bf = (
                0.5 * np.log(var_e / (var_g * lhs + 1e-30))
                + 0.5 * mean_j ** 2 / var_j
            )
            log_odds = np.log(max(1 - pi, 1e-12)) - np.log(max(pi, 1e-12)) + log_bf
            prob_in = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -500, 500)))

            if rng.uniform() < prob_in:
                delta[j] = 1
                g[j] = rng.normal(mean_j, np.sqrt(max(var_j, 1e-12)))
            else:
                delta[j] = 0
                g[j] = 0.0
            e -= Z[:, j] * g[j]

        n_incl = int(np.sum(delta))
        pi = rng.beta(1.0 + p - n_incl, 1.0 + n_incl)

        if n_incl > 0:
            ss_g = np.sum(g[delta == 1] ** 2)
            shape_g = (df_prior + n_incl) / 2.0
            scale_g = (df_prior * scale_prior + ss_g) / 2.0
            var_g = scale_g / rng.gamma(shape_g)

        shape_e = (n - 2.0) / 2.0
        scale_e = np.sum(e ** 2) / 2.0
        if shape_e > 0:
            var_e = scale_e / rng.gamma(shape_e)

        if it >= burn_in:
            effects_accum += g
            pi_accum += pi
            n_saved += 1

    effects = effects_accum / max(n_saved, 1)
    pi_est = pi_accum / max(n_saved, 1)
    y_hat = Z @ effects
    corr = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0

    return GenomicsResult(
        name="BayesCpi",
        statistic=corr,
        n=n,
        extra={"effects": effects.tolist(), "pi_est": float(pi_est)},
    )


def cheatsheet() -> str:
    return "bycpi(y, Z) -> BayesCpi genomic prediction (common variance mixture)."
