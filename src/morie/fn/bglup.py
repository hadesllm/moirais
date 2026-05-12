# morie.fn — function file (hadesllm/morie)
"""BayesCπ: spike-and-slab variable selection for genomic prediction."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["bayes_cpi_genomic"]


def bayes_cpi_genomic(x, y, n_iter: int = 300, burn: int = 100,
                      pi_init: float = 0.1, seed: int = 0):
    """BayesCπ — Habier, Fernando & Garrick (2011).

    Model::

        beta_j ~ pi * N(0, sigma_b^2) + (1 - pi) * delta(0)
        pi      ~ Beta(1, 1)
        sigma_b^2, sigma^2 ~ scaled-inv-chi^2(df, S)

    Gibbs sampler with single-site updates for (delta_j, beta_j). Short
    fixed-iteration chain by default.

    Parameters
    ----------
    x : array-like (n, p)
    y : array-like (n,)
    n_iter, burn : int
    pi_init : float in (0,1). Initial inclusion probability.
    seed : int

    Returns
    -------
    RichResult with payload keys estimate, beta, beta_pip (posterior
    inclusion prob), pi, sigma_b2, sigma2, n_iter, n, p, method.

    References
    ----------
    Habier, D., Fernando, R. L., Kizilkaya, K., & Garrick, D. J. (2011).
        Extension of the Bayesian alphabet for genomic selection.
        BMC Bioinformatics, 12, 186.
    Montesinos Lopez et al. (2022), Ch. 4.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    y_mean = float(np.mean(y)); yc = y - y_mean
    Xc = X - X.mean(axis=0)
    var_y = float(np.var(yc, ddof=1)) if n > 1 else 1.0
    sigma_b2 = var_y / max(p, 1)
    sigma2 = var_y
    pi_in = float(pi_init)
    df_b, S_b = 4.0, max(sigma_b2 * (4.0 - 2.0) / 4.0, 1e-6)
    df_e, S_e = 4.0, max(var_y * (4.0 - 2.0) / 4.0, 1e-6)

    beta = np.zeros(p)
    delta = np.zeros(p, dtype=int)
    r = yc.copy()
    xtx_diag = np.einsum("ij,ij->j", Xc, Xc)
    pip_chain = []
    beta_chain = []
    pi_chain = []
    sigma_b2_chain = []
    sigma2_chain = []

    for it in range(n_iter):
        for j in range(p):
            xj = Xc[:, j]
            # Adjust working residual for current j
            r_j = r + xj * beta[j]
            v = xtx_diag[j] / sigma2 + 1.0 / sigma_b2
            mean_j = (xj @ r_j) / sigma2 / v
            # Bayes factor for delta=1 vs delta=0
            # Marginal likelihood ratio:
            #   log BF = 0.5 * log(1/(sigma_b2*v)) + 0.5 * v * mean_j^2
            log_bf = 0.5 * np.log(1.0 / max(sigma_b2 * v, 1e-30)) + 0.5 * v * mean_j ** 2
            log_pi = np.log(max(pi_in, 1e-30))
            log_1mpi = np.log(max(1 - pi_in, 1e-30))
            log_p1 = log_pi + log_bf
            log_p0 = log_1mpi
            m = max(log_p1, log_p0)
            prob_in = np.exp(log_p1 - m) / (np.exp(log_p1 - m) + np.exp(log_p0 - m))
            delta[j] = int(rng.uniform() < prob_in)
            if delta[j]:
                beta[j] = rng.normal(mean_j, 1.0 / np.sqrt(v))
            else:
                beta[j] = 0.0
            r = r_j - xj * beta[j]
        # Update pi ~ Beta(1 + k_in, 1 + p - k_in)
        k_in = int(delta.sum())
        pi_in = float(rng.beta(1 + k_in, 1 + p - k_in))
        # Update sigma_b2 (scaled-inv-chi^2)
        df_post = df_b + max(k_in, 1)
        scale_post = (S_b * df_b + np.sum(beta[delta == 1] ** 2)) / df_post
        sigma_b2 = scale_post * df_post / max(rng.chisquare(df_post), 1e-8)
        sigma_b2 = max(sigma_b2, 1e-12)
        # Update sigma^2 (scaled-inv-chi^2)
        df_post_e = df_e + n
        scale_post_e = (S_e * df_e + np.sum(r ** 2)) / df_post_e
        sigma2 = scale_post_e * df_post_e / max(rng.chisquare(df_post_e), 1e-8)
        sigma2 = max(sigma2, 1e-12)
        if it >= burn:
            pip_chain.append(delta.copy())
            beta_chain.append(beta.copy())
            pi_chain.append(pi_in)
            sigma_b2_chain.append(sigma_b2)
            sigma2_chain.append(sigma2)
    B = np.array(beta_chain)
    PIP = np.array(pip_chain)
    beta_hat = B.mean(axis=0)
    pip = PIP.mean(axis=0)
    pi_hat = float(np.mean(pi_chain))
    sigma_b2_hat = float(np.mean(sigma_b2_chain))
    sigma2_hat = float(np.mean(sigma2_chain))
    return RichResult(
        title="BayesCπ (spike-and-slab)",
        summary_lines=[
            ("n", n),
            ("p", p),
            ("n_iter (after burn)", len(beta_chain)),
            ("posterior pi", pi_hat),
            ("posterior sigma_b^2", sigma_b2_hat),
            ("posterior sigma^2", sigma2_hat),
            ("mean PIP", float(np.mean(pip))),
        ],
        payload={
            "estimate": float(np.mean(np.abs(beta_hat))),
            "beta": beta_hat,
            "beta_pip": pip,
            "pi": pi_hat,
            "sigma_b2": sigma_b2_hat,
            "sigma2": sigma2_hat,
            "intercept": y_mean,
            "n_iter": len(beta_chain),
            "n": n,
            "p": p,
            "method": "BayesCπ Gibbs (Habier-Fernando-Kizilkaya-Garrick)",
        },
        warnings=["Short chain (default 300 iters / 100 burn) — for "
                  "publication posteriors use BGLR with ≥10k iters."],
    )


def cheatsheet():
    return "bglup: BayesCπ for variable selection"


# CANONICAL TEST
# np.random.seed(11); X = np.random.randn(30, 6); beta_true = np.array([1,0,0,-1,0,0])
# y = X @ beta_true + 0.1*np.random.randn(30)
# r = bayes_cpi_genomic(X, y, seed=11); high PIP on indices 0,3.
