# morie.fn — function file (hadesllm/morie)
"""Markov-switching regression (Hamilton 1989)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["regime_switching"]


def regime_switching(x, k_regimes=2):
    r"""Fit a constant-mean, switching-variance Markov-switching model.

    .. math::

        y_t = \mu_{s_t} + \epsilon_t,\quad
        \epsilon_t\sim\mathcal{N}(0, \sigma_{s_t}^2),\quad
        s_t\in\{1,\dots,K\}\ \text{Markov chain}.

    Estimated by EM (Hamilton filter) when statsmodels is unavailable.

    Parameters
    ----------
    x : array-like
        Univariate time series.
    k_regimes : int, default 2
        Number of latent regimes.

    Returns
    -------
    RichResult
        keys: ``mu`` (length K), ``sigma`` (length K), ``transition``
        (K × K matrix), ``smoothed_probabilities`` (n × K), ``loglik``,
        ``n``, ``method``.

    References
    ----------
    Hamilton JD (1989). A New Approach to the Economic Analysis of
    Non-Stationary Time Series and the Business Cycle. *Econometrica*
    57(2), 357-384.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < 4 * k_regimes:
        raise ValueError(f"Need at least 4*K obs, got {n}, K={k_regimes}.")

    try:
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression,
        )
        mod = MarkovRegression(y, k_regimes=k_regimes,
                               switching_variance=True)
        fit = mod.fit(disp=False)
        mu = np.asarray([float(fit.params[f"const[{k}]"])
                         for k in range(k_regimes)])
        sigma = np.asarray([float(np.sqrt(fit.params[f"sigma2[{k}]"]))
                            for k in range(k_regimes)])
        P = np.asarray(fit.regime_transition).reshape(k_regimes, k_regimes)
        return RichResult(payload={
            "mu": mu, "sigma": sigma,
            "transition": P,
            "smoothed_probabilities": np.asarray(fit.smoothed_marginal_probabilities),
            "loglik": float(fit.llf),
            "n": int(n), "k_regimes": int(k_regimes),
            "method": f"MarkovRegression via statsmodels (K={k_regimes})",
        })
    except Exception:
        pass

    # ---- Pure-NumPy EM with Hamilton filter ------------------------------
    rng = np.random.default_rng(0)
    mu = np.linspace(y.min(), y.max(), k_regimes)
    sig = np.full(k_regimes, max(y.std(), 1e-6))
    P = np.full((k_regimes, k_regimes), 1.0 / k_regimes)
    pi = np.full(k_regimes, 1.0 / k_regimes)
    ll_prev = -np.inf
    for _ in range(200):
        # E-step: forward-backward filter.
        emit = np.exp(-0.5 * ((y[:, None] - mu) / sig) ** 2) / (
            sig * np.sqrt(2 * np.pi))
        emit = np.clip(emit, 1e-300, None)
        alpha = np.zeros((n, k_regimes))
        c = np.zeros(n)
        alpha[0] = pi * emit[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ P) * emit[t]
            c[t] = alpha[t].sum()
            alpha[t] /= max(c[t], 1e-300)
        beta = np.zeros((n, k_regimes))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = P @ (emit[t + 1] * beta[t + 1])
            beta[t] /= max(beta[t].sum(), 1e-300)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((n - 1, k_regimes, k_regimes))
        for t in range(n - 1):
            xi[t] = (alpha[t, :, None] * P * emit[t + 1, None, :]
                     * beta[t + 1, None, :])
            xi[t] /= max(xi[t].sum(), 1e-300)
        # M-step.
        pi = gamma[0]
        P = xi.sum(axis=0) / np.clip(gamma[:-1].sum(axis=0)[:, None], 1e-12, None)
        for k in range(k_regimes):
            wk = gamma[:, k]
            mu[k] = np.sum(wk * y) / max(wk.sum(), 1e-12)
            sig[k] = np.sqrt(np.sum(wk * (y - mu[k]) ** 2)
                              / max(wk.sum(), 1e-12))
            sig[k] = max(sig[k], 1e-6)
        ll = float(np.sum(np.log(np.clip(c, 1e-300, None))))
        if abs(ll - ll_prev) < 1e-6:
            break
        ll_prev = ll
    return RichResult(payload={
        "mu": mu, "sigma": sig,
        "transition": P,
        "smoothed_probabilities": gamma,
        "loglik": float(ll_prev),
        "n": int(n), "k_regimes": int(k_regimes),
        "method": f"Markov switching via EM/Hamilton filter (K={k_regimes}, numpy)",
    })


def cheatsheet():
    return "regms: Markov-switching regression (Hamilton 1989)."
