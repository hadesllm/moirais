# morie.fn — function file (hadesllm/morie)
"""Hierarchical Bayes nonparametric — alpha-integrated DP."""
import numpy as np
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist, norm
from ._richresult import RichResult

__all__ = ["ghosal_hierarchical_bayes"]


def ghosal_hierarchical_bayes(x, a_prior=1.0, b_prior=1.0, M=400, seed=0,
                                deterministic_seed: int | None = None):
    """Hierarchical-DP with a Gamma(a, b) hyperprior on the concentration.

    Model::
        theta_i | G   ~ G,
        G | alpha     ~ DP(alpha, G_0),
        alpha          ~ Gamma(a, b)              (Escobar–West 1995).

    Conditional on the observed distinct-value count ``K_n``, the
    posterior of ``alpha`` is the mixture (Escobar & West 1995 eq. 13)

        p(alpha | K_n, n) prop  alpha^{K_n - 1} (alpha + n) Beta(alpha+1, n) p(alpha)

    sampled here as a 2-step augmentation:
      1. draw ``eta | alpha, n ~ Beta(alpha+1, n)``;
      2. draw ``alpha | eta, K_n`` from a mixture of two Gammas.

    Parameters
    ----------
    x : array-like — sample.
    a_prior, b_prior : Gamma(shape, rate) hyperprior on alpha.
    M : int — number of posterior draws.
    seed : int.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with ``estimate`` (posterior-mean alpha),
    ``alpha_draws``, ``alpha_se``, ``K_n``.

    References
    ----------
    Escobar & West (1995). Bayesian Density Estimation via DP-mixtures.
      JASA 90.
    Ghosal & van der Vaart (2017) Ch 15.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ghhbp", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "estimate": float("nan"), "n": n,
            "method": "Hierarchical NP-Bayes (n<2)",
        })
    K_n = int(np.unique(x).size)
    if K_n == n:
        K_n = max(2, int(np.ceil(np.log2(n) + 1)))
    a = float(a_prior)
    b = float(b_prior)
    alpha = 1.0
    draws = np.empty(M)
    for m in range(M):
        eta = rng.beta(alpha + 1.0, n)
        # mixing weight pi_eta
        w1 = a + K_n - 1
        w2 = n * (b - np.log(eta))
        p_eta = w1 / (w1 + w2)
        if rng.uniform() < p_eta:
            alpha = rng.gamma(shape=a + K_n, scale=1.0 / (b - np.log(eta)))
        else:
            alpha = rng.gamma(shape=a + K_n - 1, scale=1.0 / (b - np.log(eta)))
        draws[m] = alpha
    burn = M // 4
    chain = draws[burn:]
    return RichResult(payload={
        "estimate": float(np.mean(chain)),
        "alpha_se": float(np.std(chain, ddof=1)),
        "alpha_draws": chain.tolist(),
        "K_n": int(K_n),
        "n": n,
        "method": "Escobar-West augmentation for alpha | K_n",
    })


def cheatsheet():
    return "ghhbp: Hierarchical Bayes nonparametric"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghhbp import ghosal_hierarchical_bayes
# >>> r = ghosal_hierarchical_bayes(np.random.default_rng(0).normal(size=50))
# >>> r["estimate"] > 0
# True
