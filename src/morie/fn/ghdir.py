# morie.fn — function file (hadesllm/morie)
"""Dirichlet process posterior update."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_dirichlet_posterior"]


def ghosal_dirichlet_posterior(x, alpha=1.0, base_mean=0.0, base_sd=1.0,
                                grid=None):
    """Dirichlet-process posterior, conjugate update.

    Given prior ``G ~ DP(alpha, G0)`` with base measure ``G0 = N(base_mean,
    base_sd^2)``, the posterior given an i.i.d. sample
    ``X_1,...,X_n`` is

        G | X_{1:n} ~ DP( alpha + n,  (alpha G0 + sum_i delta_{X_i}) / (alpha+n) )

    The posterior predictive CDF evaluated on a grid ``t`` is the
    posterior mean

        F_n(t) = E[G((-inf,t]) | X_{1:n}]
               = (alpha G0(t) + sum_i 1{X_i <= t}) / (alpha + n).

    The posterior variance at each grid point (Ferguson 1973) is

        Var[G(A) | X_{1:n}] = F_n(t)(1 - F_n(t)) / (alpha + n + 1).

    Parameters
    ----------
    x : array-like
        Observations.
    alpha : float, default 1.0
        DP concentration parameter.
    base_mean, base_sd : float
        Base measure ``G0 = N(base_mean, base_sd^2)``.
    grid : array-like or None
        Points at which to evaluate the posterior predictive CDF.
        If ``None``, an evenly spaced 51-point grid spanning the
        observed data is used.

    Returns
    -------
    RichResult with keys ``estimate`` (posterior-mean CDF value at the
    sample mean), ``cdf_grid``, ``cdf_post``, ``cdf_var``,
    ``alpha_post``, ``n``.

    References
    ----------
    Ferguson, T. (1973). A Bayesian Analysis of Some Nonparametric
      Problems. Annals of Statistics 1(2).
    Ghosal & van der Vaart (2017) Ch 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if grid is None:
        if n == 0:
            grid = np.linspace(base_mean - 3 * base_sd,
                               base_mean + 3 * base_sd, 51)
        else:
            lo, hi = float(np.min(x)), float(np.max(x))
            pad = max(1e-6, 0.1 * (hi - lo + 1.0))
            grid = np.linspace(lo - pad, hi + pad, 51)
    grid = np.asarray(grid, dtype=float)
    alpha_post = float(alpha + n)
    G0_t = norm.cdf(grid, loc=base_mean, scale=base_sd)
    if n > 0:
        emp_t = np.array([(x <= t).sum() for t in grid], dtype=float)
    else:
        emp_t = np.zeros_like(grid)
    F_post = (alpha * G0_t + emp_t) / alpha_post
    var_post = F_post * (1.0 - F_post) / (alpha_post + 1.0)
    # Headline scalar: posterior CDF at the sample mean (a stable single
    # number for the regression/agreement tests).
    if n > 0:
        t0 = float(np.mean(x))
        G0_t0 = float(norm.cdf(t0, loc=base_mean, scale=base_sd))
        emp_t0 = float((x <= t0).mean()) * n
        estimate = (alpha * G0_t0 + emp_t0) / alpha_post
    else:
        estimate = float(norm.cdf(base_mean, loc=base_mean, scale=base_sd))
    return RichResult(payload={
        "estimate": float(estimate),
        "alpha_post": alpha_post,
        "n": n,
        "cdf_grid": grid.tolist(),
        "cdf_post": F_post.tolist(),
        "cdf_var": var_post.tolist(),
        "method": "Dirichlet process posterior (conjugate)",
    })


def cheatsheet():
    return "ghdir: Dirichlet process posterior update"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghdir import ghosal_dirichlet_posterior
# >>> rng = np.random.default_rng(0)
# >>> r = ghosal_dirichlet_posterior(rng.normal(size=100), alpha=2.0)
# >>> 0.0 < r.estimate < 1.0
# True
