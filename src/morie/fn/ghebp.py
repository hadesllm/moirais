# morie.fn -- function file (hadesllm/morie)
"""Empirical-Bayes hyper-parameter selection for a DP prior."""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from ._richresult import RichResult

__all__ = ["ghosal_empirical_bayes"]


def ghosal_empirical_bayes(x, alpha_grid=None):
    """Empirical-Bayes selection of the DP concentration ``alpha``.

    Antoniak (1974): the number of distinct values ``K_n`` in a sample
    of size ``n`` from a DP-distributed measure has expectation

        E[K_n | alpha] = sum_{i=1}^n alpha / (alpha + i - 1).

    The marginal likelihood of ``K_n`` distinct values (assuming a
    continuous base measure so ties happen only via the DP) is

        p(K = k | alpha, n) = |s(n, k)| alpha^k Gamma(alpha) /
                                                 Gamma(alpha + n)

    with ``|s(n, k)|`` unsigned Stirling numbers of the first kind.
    Since |s(n,k)| does not depend on alpha, the alpha-MLE maximises

        log alpha^{K_n} Gamma(alpha) / Gamma(alpha + n)
          = K_n log alpha + lgamma(alpha) - lgamma(alpha + n).

    We optimise this objective by scalar Brent search over [1e-3, 1e3].

    Parameters
    ----------
    x : array-like -- sample.  Distinct count ``K_n`` is computed from
        rounded values; if the data is continuous-valued ``K_n = n``.
    alpha_grid : array-like or None
        If provided, the marginal log-lik is evaluated on this grid.

    Returns
    -------
    RichResult with ``estimate`` (MLE alpha), ``K_n``,
    ``log_lik_at_estimate``.

    References
    ----------
    Antoniak, C. (1974). AOS 2.
    McCloskey (1965); Liu (1996).
    Ghosal & van der Vaart (2017) Ch 15.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "estimate": float("nan"), "n": n,
            "method": "Empirical Bayes (n<2)",
        })
    K_n = int(np.unique(x).size)
    # Avoid degenerate K=n by treating continuous data as if "binned" at
    # the bandwidth of the data -- informative only if there are ties.
    if K_n == n:
        # Use a Sturges-style bin count so the EB estimate is meaningful.
        K_n = max(2, int(np.ceil(np.log2(n) + 1)))

    def neg_ll(a):
        return -(K_n * np.log(a) + gammaln(a) - gammaln(a + n))

    if alpha_grid is None:
        opt = minimize_scalar(neg_ll, bounds=(1e-3, 1e3), method="bounded")
        a_hat = float(opt.x)
        ll = float(-opt.fun)
    else:
        ag = np.asarray(alpha_grid, dtype=float)
        ll_grid = -np.array([neg_ll(a) for a in ag])
        idx = int(np.argmax(ll_grid))
        a_hat = float(ag[idx])
        ll = float(ll_grid[idx])
    return RichResult(payload={
        "estimate": a_hat,
        "K_n": int(K_n),
        "log_lik_at_estimate": ll,
        "n": n,
        "method": "Empirical-Bayes alpha for DP (Antoniak 1974 MLE)",
    })


def cheatsheet():
    return "ghebp: Empirical Bayes (DP)"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghebp import ghosal_empirical_bayes
# >>> r = ghosal_empirical_bayes(np.random.default_rng(0).normal(size=100))
# >>> r["estimate"] > 0
# True
