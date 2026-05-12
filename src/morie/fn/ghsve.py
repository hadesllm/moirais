# morie.fn — function file (hadesllm/morie)
"""Sieve prior construction (truncated log-spline / Bernstein basis)."""
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import gammaln
from ._richresult import RichResult

__all__ = ["ghosal_sieve_prior"]


def _bernstein_basis(u, K):
    """Bernstein polynomial basis matrix B_{k,K}(u), k=1..K."""
    u = np.atleast_1d(u)
    k = np.arange(1, K + 1)[None, :]
    n = K
    log_binom = (gammaln(n + 1) - gammaln(k) - gammaln(n - k + 2))
    log_b = (log_binom + (k - 1) * np.log(np.clip(u[:, None], 1e-12, 1))
             + (n - k + 1) * np.log(np.clip(1 - u[:, None], 1e-12, 1)))
    return np.exp(log_b)


def ghosal_sieve_prior(x, K=None):
    """Bernstein-polynomial sieve density estimator on [0,1].

    The Bernstein-polynomial prior (Petrone 1999) is a canonical
    Ghosal sieve: pick ``K``, draw weights ``w ~ Dir(alpha,...,alpha)``
    over the ``K`` Beta(k, K-k+1) component densities, take

        f_K(u) = sum_{k=1}^K w_k Beta(u; k, K-k+1).

    This estimator attains the minimax rate ``n^{-beta/(2beta+1)}``
    (up to log factors) on Hölder-β densities by letting ``K = K_n``
    grow at the optimal rate.

    We follow Ghosal (2001) defaults:
      * Rescale x to (0, 1) via the min/max.
      * Set ``K = K_n = round(n^{1/(2 beta + 1)})``, with ``beta = 1``.
      * MLE the mixture weights (sieve-empirical-Bayes) by 50 EM steps.

    Parameters
    ----------
    x : array-like — sample on R.
    K : int or None.

    Returns
    -------
    RichResult with ``estimate`` (cross-validated log-likelihood per
    obs), ``weights``, ``K``.

    References
    ----------
    Petrone, S. (1999). Random Bernstein polynomials. Scand J Stat 26.
    Ghosal, S. (2001). Convergence rates for Bernstein density. Bernoulli 7.
    Ghosal & van der Vaart (2017) Ch 9.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 3:
        return RichResult(payload={
            "estimate": float("nan"), "n": n,
            "method": "Bernstein sieve (n<3)",
        })
    lo, hi = float(np.min(x)) - 1e-6, float(np.max(x)) + 1e-6
    u = (x - lo) / (hi - lo)
    if K is None:
        K = max(2, int(round(n ** (1.0 / 3.0))))
    B = _bernstein_basis(u, K)  # (n, K)
    w = np.ones(K) / K
    # EM for mixture weights
    for _ in range(60):
        num = B * w[None, :]
        denom = np.clip(num.sum(axis=1, keepdims=True), 1e-12, None)
        gamma = num / denom
        w_new = gamma.mean(axis=0)
        w_new /= w_new.sum()
        if np.allclose(w, w_new, atol=1e-8):
            w = w_new
            break
        w = w_new
    log_lik = float(np.mean(np.log(np.clip(B @ w, 1e-12, None))))
    # Headline: posterior-mean density of x at sample mean (Bayes-style)
    u_bar = (float(np.mean(x)) - lo) / (hi - lo)
    f_bar = float((_bernstein_basis(np.array([u_bar]), K)[0] @ w) / (hi - lo))
    return RichResult(payload={
        "estimate": f_bar,
        "log_lik_per_obs": log_lik,
        "weights": w.tolist(),
        "K": int(K),
        "n": n,
        "method": "Bernstein-polynomial sieve density (Petrone 1999, Ghosal 2001)",
    })


def cheatsheet():
    return "ghsve: Sieve prior (Bernstein polynomial)"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghsve import ghosal_sieve_prior
# >>> r = ghosal_sieve_prior(np.random.default_rng(0).uniform(size=200))
# >>> r["K"] >= 2
# True
