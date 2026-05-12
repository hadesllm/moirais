# morie.fn — function file (hadesllm/morie)
"""Semiparametric mixture model via EM (Horowitz 2009, Ch 11).

Two-component Gaussian mixture (the canonical workhorse for the
semiparametric mixture chapter):

    f(y) = pi * N(y; mu1, sigma1^2) + (1 - pi) * N(y; mu2, sigma2^2)

estimated by Expectation-Maximisation with k-means warm start.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_mixture_model"]


def _gauss_pdf(y, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def horowitz_mixture_model(y, k=2, maxiter=200, tol=1e-6, seed=0):
    """Two-component (or k-component) Gaussian-mixture EM.

    Parameters
    ----------
    y : array-like
    k : int, default 2
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < max(10, 3 * k):
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "mixture-EM (insufficient data)"})
    k = max(2, int(k))
    rng = np.random.default_rng(seed)
    # k-means style warm start via quantile init
    qs = np.linspace(0.1, 0.9, k)
    mu = np.quantile(y, qs).astype(float)
    sigma = np.full(k, float(np.std(y, ddof=1) / k + 1e-3))
    pi = np.full(k, 1.0 / k)
    ll_prev = -np.inf
    for it in range(maxiter):
        # E-step
        comps = np.column_stack([pi[j] * _gauss_pdf(y, mu[j], sigma[j])
                                 for j in range(k)])
        denom = comps.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1e-12)
        gamma = comps / denom
        # M-step
        Nk = gamma.sum(axis=0)
        Nk = np.where(Nk > 0, Nk, 1e-12)
        mu = (gamma * y[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt((gamma * (y[:, None] - mu) ** 2).sum(axis=0) / Nk)
        sigma = np.maximum(sigma, 1e-4)
        pi = Nk / n
        ll = float(np.log(np.maximum(comps.sum(axis=1), 1e-300)).sum())
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll
    return RichResult(payload={
        "estimate": {"pi": pi.astype(float), "mu": mu.astype(float),
                     "sigma": sigma.astype(float)},
        "log_likelihood": ll_prev,
        "n": n, "k": k, "iters": it + 1,
        "method": f"{k}-component Gaussian mixture EM",
    })


def cheatsheet():
    return "hrzm1: Gaussian-mixture EM"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(14)
    n = 1000
    z = rng.uniform(size=n) < 0.4
    y = np.where(z, rng.normal(0, 1, n), rng.normal(5, 1, n))
    res = horowitz_mixture_model(y, k=2)
    print(res)
    mus = sorted(res["estimate"]["mu"])
    assert abs(mus[0] - 0) < 1.0 and abs(mus[1] - 5) < 1.0
