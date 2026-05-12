# morie.fn -- function file (hadesllm/morie)
"""Moment matching: posterior mean and variance of G(A) for a DP."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_moment_matching"]


def ghosal_moment_matching(x, alpha=1.0, A_lower=None, A_upper=None,
                            base_mean=0.0, base_sd=1.0):
    """Posterior mean / variance of ``G(A)`` under a DP prior.

    For ``G ~ DP(alpha, G_0)`` and any Borel set ``A``::

        E[G(A)]               = G_0(A),
        Var[G(A)]             = G_0(A) (1 - G_0(A)) / (alpha + 1),
        E[G(A) | X_{1:n}]     = (alpha G_0(A) + n_A) / (alpha + n),
        Var[G(A) | X_{1:n}]   = E[G(A)|X](1 - E[G(A)|X]) / (alpha + n + 1).

    Here ``A = (A_lower, A_upper]``.  Defaults to ``(-inf, mean(x)]``
    so the headline statistic is interpretable as a posterior CDF
    value.

    Parameters
    ----------
    x : array-like.
    alpha : float.
    A_lower, A_upper : float or None.
    base_mean, base_sd : float.

    Returns
    -------
    RichResult with ``estimate`` = E[G(A)|X], ``se`` =
    sqrt(Var[G(A)|X]), ``prior_mean``, ``prior_var``, ``n_A``.

    References
    ----------
    Ferguson, T. (1973). AOS 1.
    Ghosal & van der Vaart (2017) Ch 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if A_lower is None:
        A_lower = -np.inf
    if A_upper is None:
        A_upper = float(np.mean(x)) if n else 0.0
    G0_A = float(norm.cdf(A_upper, loc=base_mean, scale=base_sd)
                  - norm.cdf(A_lower, loc=base_mean, scale=base_sd))
    G0_A = float(np.clip(G0_A, 0, 1))
    prior_mean = G0_A
    prior_var = G0_A * (1 - G0_A) / (alpha + 1)
    n_A = int(((x > A_lower) & (x <= A_upper)).sum()) if n else 0
    post_mean = (alpha * G0_A + n_A) / (alpha + n)
    post_var = post_mean * (1 - post_mean) / (alpha + n + 1)
    return RichResult(payload={
        "estimate": float(post_mean),
        "se": float(np.sqrt(max(post_var, 0))),
        "prior_mean": prior_mean,
        "prior_var": prior_var,
        "n_A": n_A,
        "n": n,
        "alpha": float(alpha),
        "method": "DP moment-matching (Ferguson 1973)",
    })


def cheatsheet():
    return "ghmmt: Moment matching for DP"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghmmt import ghosal_moment_matching
# >>> r = ghosal_moment_matching(np.array([0., 0., 0.]), alpha=1.0,
# ...                              A_lower=-1, A_upper=1)
# >>> 0 <= r["estimate"] <= 1
# True
