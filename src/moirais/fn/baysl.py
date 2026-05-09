# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian LASSO for genomic prediction."""

__all__ = ["baysl"]

import numpy as np

from ._containers import GenomicsResult


def baysl(
    y: np.ndarray,
    Z: np.ndarray,
    *,
    n_iter: int = 1000,
    burn_in: int = 200,
    lambda_init: float = 1.0,
    seed: int = 42,
) -> GenomicsResult:
    """Bayesian LASSO for genomic prediction via Gibbs sampling.

    Places a Laplace (double-exponential) prior on marker effects,
    which is equivalent to a Gaussian scale mixture with exponential
    mixing density on the variance of each effect.

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
    lambda_init : float
        Initial regularization parameter (squared).
    seed : int
        Random seed.

    Returns
    -------
    GenomicsResult
        statistic = cor(y, y_hat), extra has 'effects'
        and 'lambda_sq' (posterior mean of lambda^2).

    References
    ----------
    Park, T., & Casella, G. (2008). The Bayesian Lasso. JASA,
        103(482), 681-686.
    de los Campos, G., et al. (2009). Predicting quantitative traits
        with regression models for dense molecular markers and
        pedigree. Genetics, 182(1), 375-385.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape

    if n != len(y):
        raise ValueError("Z rows must match length of y.")

    ztz = np.sum(Z ** 2, axis=0)
    g = np.zeros(p)
    var_e = np.var(y) * 0.5
    tau2 = np.ones(p)
    lambda_sq = lambda_init
    e = y - Z @ g

    effects_accum = np.zeros(p)
    lam_accum = 0.0
    n_saved = 0

    for it in range(n_iter):
        for j in range(p):
            e += Z[:, j] * g[j]
            rhs = Z[:, j] @ e
            lhs = ztz[j] + var_e / max(tau2[j], 1e-12)
            mean_j = rhs / lhs
            var_j = var_e / lhs
            g[j] = rng.normal(mean_j, np.sqrt(max(var_j, 1e-12)))
            e -= Z[:, j] * g[j]

        for j in range(p):
            mu_ig = np.sqrt(lambda_sq * var_e / max(g[j] ** 2, 1e-12))
            try:
                inv_tau = rng.wald(mu_ig, lambda_sq)
                tau2[j] = 1.0 / max(inv_tau, 1e-12)
            except (ValueError, ZeroDivisionError):
                tau2[j] = 1.0

        shape_lam = p + 1.0
        rate_lam = np.sum(tau2) / 2.0 + 1e-4
        lambda_sq = rng.gamma(shape_lam, 1.0 / rate_lam)

        shape_e = (n - 2.0) / 2.0
        scale_e = np.sum(e ** 2) / 2.0
        if shape_e > 0:
            var_e = scale_e / rng.gamma(shape_e)

        if it >= burn_in:
            effects_accum += g
            lam_accum += lambda_sq
            n_saved += 1

    effects = effects_accum / max(n_saved, 1)
    lam_est = lam_accum / max(n_saved, 1)
    y_hat = Z @ effects
    corr = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else 0.0

    return GenomicsResult(
        name="BayesianLASSO",
        statistic=corr,
        n=n,
        extra={"effects": effects.tolist(), "lambda_sq": float(lam_est)},
    )


def cheatsheet() -> str:
    return "baysl(y, Z) -> Bayesian LASSO for genomic prediction."
