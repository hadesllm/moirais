# morie.fn — function file (hadesllm/morie)
"""DP-mixture density estimate (collapsed Gibbs)."""
import numpy as np
from scipy.stats import norm, t as student_t
from ._richresult import RichResult

__all__ = ["ghosal_dpmixture_density"]


def ghosal_dpmixture_density(x, alpha=1.0, sigma=None, grid=None,
                              n_iter=120, burn=40, seed=0,
                              deterministic_seed: int | None = None):
    """Posterior-mean density under a DP mixture of normals.

    Model::
        X_i | mu_i      ~ N(mu_i, sigma^2)            (sigma fixed/plug-in)
        mu_i | G        ~ G
        G               ~ DP(alpha, G0),  G0 = N(m0, s0^2)

    Implements Algorithm 3 of Neal (2000) — a collapsed Polya-urn Gibbs
    sampler on cluster labels with marginalised cluster means.  The
    posterior-mean predictive density evaluated on a grid ``t`` is

        f_hat(t) = (1/M) sum_{m} sum_{k} (n_k^{(m)}/(alpha+n)) phi_t(t; mu_k^{(m)})
                                       + (alpha/(alpha+n)) E_{G0}[phi_t]

    where the second term is the prior-predictive contribution.

    Parameters
    ----------
    x : array-like
    alpha : float
        DP concentration.
    sigma : float or None
        Within-cluster sd (Silverman bandwidth if None).
    grid : array-like or None
        Evaluation grid (51 pts spanning the data ±1 sd if None).
    n_iter, burn : int
        MCMC iterations.
    seed : int
        RNG seed.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged: the user-supplied ``seed`` drives a fresh
        :class:`numpy.random.Generator`.

    Returns
    -------
    RichResult with ``estimate`` (mean over the grid),
    ``density``, ``grid``, ``k_post`` (posterior-mean #clusters).

    References
    ----------
    Neal, R. (2000). Markov-Chain Sampling for DP-Mixture Models. JCGS 9.
    Escobar & West (1995). Bayesian Density Estimation. JASA 90.
    Ghosal & van der Vaart (2017) Ch 4.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ghdpm", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return RichResult(payload={
            "estimate": float("nan"), "n": 0,
            "method": "DP-mixture density (empty input)",
        })
    if sigma is None:
        # Silverman's rule of thumb bandwidth as the within-cluster sd
        s = float(np.std(x, ddof=1)) if n > 1 else 1.0
        sigma = 1.06 * max(s, 1e-6) * n ** (-1.0 / 5.0)
    sigma = float(max(sigma, 1e-6))
    m0 = float(np.mean(x))
    s0 = float(np.std(x, ddof=1)) if n > 1 else 1.0
    s0 = max(s0, 1e-3)
    if grid is None:
        lo, hi = float(np.min(x)) - s0, float(np.max(x)) + s0
        grid = np.linspace(lo, hi, 51)
    grid = np.asarray(grid, dtype=float)

    # Conjugate predictive for a new cluster: marginalising mu out of
    # N(mu, sigma^2) with mu ~ N(m0, s0^2) yields N(m0, sigma^2 + s0^2).
    def new_cluster_logp(xi):
        return norm.logpdf(xi, loc=m0, scale=np.sqrt(sigma ** 2 + s0 ** 2))

    # Posterior of mu_k given members x_S of cluster k:
    #   mu_k | x_S ~ N(m_post, v_post)
    #   v_post^{-1} = 1/s0^2 + |S|/sigma^2
    #   m_post      = v_post (m0/s0^2 + sum_S x / sigma^2)
    def cluster_post(xs):
        nk = len(xs)
        v = 1.0 / (1.0 / s0 ** 2 + nk / sigma ** 2)
        m = v * (m0 / s0 ** 2 + np.sum(xs) / sigma ** 2)
        return m, v

    # Predictive density of x_i given cluster k (marginal over mu_k)
    def in_cluster_logp(xi, xs):
        m, v = cluster_post(xs)
        return norm.logpdf(xi, loc=m, scale=np.sqrt(v + sigma ** 2))

    labels = np.zeros(n, dtype=int)
    k_chain = []
    f_chain = []
    for it in range(n_iter):
        for i in range(n):
            # remove i
            old = labels[i]
            labels[i] = -1
            uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
            log_probs = []
            for k, ck in zip(uniq, counts):
                xs = x[(labels == k)]
                log_probs.append(np.log(ck) + in_cluster_logp(x[i], xs))
            log_probs.append(np.log(alpha) + new_cluster_logp(x[i]))
            log_probs = np.asarray(log_probs)
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            choice = rng.choice(len(probs), p=probs)
            if choice == len(uniq):
                new_label = int(uniq.max() + 1) if len(uniq) else 0
                labels[i] = new_label
            else:
                labels[i] = int(uniq[choice])
        if it >= burn:
            uniq = np.unique(labels)
            f = np.zeros_like(grid)
            for k in uniq:
                xs = x[labels == k]
                m, v = cluster_post(xs)
                f += (len(xs) / (alpha + n)) * norm.pdf(
                    grid, loc=m, scale=np.sqrt(v + sigma ** 2))
            f += (alpha / (alpha + n)) * norm.pdf(
                grid, loc=m0, scale=np.sqrt(sigma ** 2 + s0 ** 2))
            f_chain.append(f)
            k_chain.append(len(uniq))

    density = np.mean(np.asarray(f_chain), axis=0)
    estimate = float(np.trapezoid(density * grid, grid) /
                     max(np.trapezoid(density, grid), 1e-12))
    return RichResult(payload={
        "estimate": estimate,
        "grid": grid.tolist(),
        "density": density.tolist(),
        "k_post": float(np.mean(k_chain)),
        "n": n,
        "alpha": float(alpha),
        "sigma": float(sigma),
        "method": "DP-mixture density via collapsed Gibbs (Neal 2000 Alg 3)",
    })


def cheatsheet():
    return "ghdpm: DP mixture density estimate"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghdpm import ghosal_dpmixture_density
# >>> rng = np.random.default_rng(0)
# >>> r = ghosal_dpmixture_density(rng.normal(size=50), n_iter=20, burn=5)
# >>> r["k_post"] >= 1
# True
