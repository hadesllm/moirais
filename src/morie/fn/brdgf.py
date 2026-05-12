# morie.fn -- function file (hadesllm/morie)
"""BayesA via Gibbs sampler (per-marker variance scaled-inverse-chi^2)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["bayes_ridge_gibbs"]


def bayes_ridge_gibbs(x, y, n_iter: int = 200, burn: int = 50,
                      df0: float = 4.0, S0: float | None = None, seed: int = 0,
                      deterministic_seed: int | None = None):
    """BayesA -- Meuwissen, Hayes & Goddard (2001).

    Model::

        y = X*beta + e,  beta_j ~ N(0, sigma_j^2),
        sigma_j^2 ~ scaled-inv-chi^2(df0, S0)

    Each marker has its own variance, sampled via Gibbs. A short fixed-iteration
    chain is run for sub-second runtime; switch to BGLR for production posteriors.

    Parameters
    ----------
    x : array-like (n, p)
    y : array-like (n,)
    n_iter, burn : int
    df0 : float, default 4
    S0 : float, optional. Prior scale; default anchors prior mode to var(y)/p.
    seed : int
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with payload keys estimate, beta, beta_se, sigma_j2, sigma2,
    n_iter, n, p, method.

    References
    ----------
    Meuwissen, T. H. E., Hayes, B. J., & Goddard, M. E. (2001). Prediction
        of total genetic value using genome-wide dense marker maps.
        Genetics, 157(4), 1819-1829.
    Montesinos Lopez et al. (2022), Ch. 4.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("brdgf", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    y_mean = float(np.mean(y))
    yc = y - y_mean
    x_mean = X.mean(axis=0)
    Xc = X - x_mean

    var_y = float(np.var(yc, ddof=1)) if n > 1 else 1.0
    if S0 is None:
        S0 = (var_y / max(p, 1)) * (df0 - 2.0) / df0
        S0 = max(S0, 1e-6)

    beta = np.zeros(p)
    sigma_j2 = np.full(p, var_y / max(p, 1))
    sigma2 = var_y

    xtx_diag = np.einsum("ij,ij->j", Xc, Xc)
    resid = yc - Xc @ beta

    beta_chain = []
    sj_chain = []
    sig_chain = []
    for it in range(n_iter):
        for j in range(p):
            xj = Xc[:, j]
            resid_j = resid + xj * beta[j]
            v = xtx_diag[j] / sigma2 + 1.0 / sigma_j2[j]
            mean_j = (xj @ resid_j) / sigma2 / v
            sd_j = 1.0 / np.sqrt(v)
            beta[j] = rng.normal(mean_j, sd_j)
            resid = resid_j - xj * beta[j]
        df_post = df0 + 1.0
        scale_post = (S0 * df0 + beta ** 2) / df_post
        chi2_draws = rng.chisquare(df_post, size=p)
        sigma_j2 = scale_post * df_post / np.maximum(chi2_draws, 1e-8)
        sigma_j2 = np.maximum(sigma_j2, 1e-12)
        df_e = 4.0
        Se = var_y * (df_e - 2.0) / df_e
        df_post_e = n + df_e
        scale_post_e = (np.sum(resid ** 2) + df_e * Se) / df_post_e
        sigma2 = scale_post_e * df_post_e / max(rng.chisquare(df_post_e), 1e-8)
        sigma2 = max(sigma2, 1e-12)
        if it >= burn:
            beta_chain.append(beta.copy())
            sj_chain.append(sigma_j2.copy())
            sig_chain.append(sigma2)
    B = np.array(beta_chain)
    SJ = np.array(sj_chain)
    beta_hat = B.mean(axis=0)
    beta_se = B.std(axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(p)
    sigma_j2_hat = SJ.mean(axis=0)
    sigma2_hat = float(np.mean(sig_chain))
    return RichResult(
        title="BayesA (per-marker variance, Gibbs)",
        summary_lines=[
            ("n", n),
            ("p", p),
            ("n_iter (after burn)", len(beta_chain)),
            ("posterior mean sigma^2", sigma2_hat),
            ("mean sigma_j^2", float(np.mean(sigma_j2_hat))),
            ("mean |beta|", float(np.mean(np.abs(beta_hat)))),
        ],
        payload={
            "estimate": float(np.mean(np.abs(beta_hat))),
            "beta": beta_hat,
            "beta_se": beta_se,
            "se": float(np.mean(beta_se)),
            "sigma_j2": sigma_j2_hat,
            "sigma2": sigma2_hat,
            "intercept": y_mean,
            "n_iter": len(beta_chain),
            "n": n,
            "p": p,
            "method": "BayesA (Meuwissen-Hayes-Goddard) short Gibbs",
        },
        warnings=["Short chain (default 200 / 50 burn) -- for publication "
                  "posteriors use BGLR with ≥10k iters."],
    )


def cheatsheet():
    return "brdgf: BayesA via Gibbs sampler"


# CANONICAL TEST
# np.random.seed(4); X = np.random.randn(20, 5); beta_true = np.array([1,-1,0.5,0,0])
# y = X @ beta_true + 0.2*np.random.randn(20)
# r = bayes_ridge_gibbs(X, y, seed=4); r.beta within 30% of beta_true.
