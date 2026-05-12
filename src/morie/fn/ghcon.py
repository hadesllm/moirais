# morie.fn — function file (hadesllm/morie)
"""Schwartz posterior consistency diagnostic."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_posterior_consistency"]


def ghosal_posterior_consistency(x, ref_loc=None, ref_scale=None,
                                  eps=0.1, K=200, seed=0):
    """Schwartz-style posterior-consistency diagnostic.

    For a Dirichlet-process prior the posterior is strongly consistent
    at any continuous distribution ``F0`` in the weak topology.  This
    function returns a Monte-Carlo estimate of

        Pi( || F - F_ref ||_KS > eps  |  X_{1:n} )

    where ``F_ref`` is either the empirical CDF (default) or a reference
    Gaussian if ``ref_loc``/``ref_scale`` are given.  Posterior samples
    are obtained via the Bayesian bootstrap (Rubin 1981) — the
    no-base-measure limit of a posterior DP.  Schwartz's theorem
    (Ghosal Ch 5) implies this probability decays to 0 as ``n -> inf``.

    Parameters
    ----------
    x : array-like
    ref_loc, ref_scale : float or None
        Reference Gaussian.  If both None, the empirical CDF is used.
    eps : float
    K : int
        Number of posterior draws.
    seed : int

    Returns
    -------
    RichResult with ``estimate`` (Pi(d > eps | data)),
    ``ks_mean``, ``ks_se``, ``schwartz_bound`` = exp(-2 n eps^2).

    References
    ----------
    Schwartz, L. (1965). Z. Wahrscheinlichkeitstheorie 4.
    Rubin, D. B. (1981). The Bayesian Bootstrap. AOS 9.
    Ghosal & van der Vaart (2017) Ch 5.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return RichResult(payload={
            "estimate": float("nan"), "n": 0,
            "method": "Schwartz consistency (empty input)",
        })
    order = np.argsort(x)
    xs = x[order]
    grid = np.linspace(float(xs[0]) - 1.0, float(xs[-1]) + 1.0, 200)
    if ref_loc is None or ref_scale is None:
        F_ref = np.searchsorted(xs, grid, side="right") / n
    else:
        F_ref = norm.cdf(grid, loc=ref_loc, scale=ref_scale)
    ks = np.empty(K)
    for k in range(K):
        u = rng.dirichlet(np.ones(n))
        cdf_at_data = np.cumsum(u)  # sorted weights cumulative
        idx = np.searchsorted(xs, grid, side="right")
        F_draw = np.where(idx == 0, 0.0,
                          cdf_at_data[np.clip(idx - 1, 0, n - 1)])
        ks[k] = float(np.max(np.abs(F_draw - F_ref)))
    estimate = float(np.mean(ks > eps))
    ks_mean = float(np.mean(ks))
    ks_se = float(np.std(ks, ddof=1) / np.sqrt(K)) if K > 1 else float("nan")
    schwartz_bound = float(np.exp(-2 * n * eps ** 2))
    return RichResult(payload={
        "estimate": estimate,
        "ks_mean": ks_mean,
        "ks_se": ks_se,
        "schwartz_bound": schwartz_bound,
        "n": n,
        "eps": float(eps),
        "method": "Schwartz consistency (Bayesian-bootstrap proxy)",
    })


def cheatsheet():
    return "ghcon: Posterior consistency (Schwartz)"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghcon import ghosal_posterior_consistency
# >>> r = ghosal_posterior_consistency(np.random.default_rng(0).normal(size=200), seed=0)
# >>> 0.0 <= r.estimate <= 1.0
# True
