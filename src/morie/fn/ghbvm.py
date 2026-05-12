# morie.fn — function file (hadesllm/morie)
"""Semiparametric Bernstein–von Mises diagnostic."""
import numpy as np
from scipy.stats import norm, kstest
from ._richresult import RichResult

__all__ = ["ghosal_bernstein_von_mises"]


def ghosal_bernstein_von_mises(x, theta0=None, B=500, seed=0,
                                 deterministic_seed: int | None = None):
    """Bernstein–von Mises diagnostic for the mean functional.

    The BvM theorem (Ghosal Ch 11) says

        sqrt(n) (theta_n - theta_0)  --d-->  N(0, I^{-1})

    under the posterior law.  For the mean functional under a
    Dirichlet-process prior (Lo 1983), the posterior of theta = E_F[X]
    is asymptotically ``N(bar X_n, S_n^2 / n)``.  This callable:

      1. Generates ``B`` Bayesian-bootstrap draws of theta.
      2. Computes the standardised statistic
         ``Z_b = sqrt(n) (theta_b - bar X) / s``.
      3. KS-tests ``Z`` against ``N(0,1)`` as a BvM check.

    Parameters
    ----------
    x : array-like.
    theta0 : float or None — null mean (defaults to bar X_n).
    B : int — number of posterior draws.
    seed : int.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with ``estimate`` (posterior-mean theta), ``se``
    (posterior sd), ``z_ks_stat``, ``z_ks_pvalue``.

    References
    ----------
    Lo, A. (1983). Bayesian Bootstrap clones. AOS 11.
    Castillo & Nickl (2014). Nonparametric BvM. AOS 42.
    Ghosal & van der Vaart (2017) Ch 11.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ghbvm", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "estimate": float("nan"), "se": float("nan"), "n": n,
            "method": "BvM (n<2)",
        })
    theta_hat = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    theta_draws = np.empty(B)
    for b in range(B):
        u = rng.dirichlet(np.ones(n))
        theta_draws[b] = float(u @ x)
    theta_mean = float(np.mean(theta_draws))
    theta_sd = float(np.std(theta_draws, ddof=1))
    z = (theta_draws - theta_hat) * np.sqrt(n) / max(s, 1e-12)
    ks = kstest(z, "norm")
    if theta0 is not None:
        wald = (theta_mean - float(theta0)) / max(theta_sd, 1e-12)
        wald_p = 2 * (1 - norm.cdf(abs(wald)))
    else:
        wald = float("nan")
        wald_p = float("nan")
    return RichResult(payload={
        "estimate": theta_mean,
        "se": theta_sd,
        "theta_hat": theta_hat,
        "z_ks_stat": float(ks.statistic),
        "z_ks_pvalue": float(ks.pvalue),
        "wald": float(wald),
        "wald_pvalue": float(wald_p),
        "n": n,
        "B": int(B),
        "method": "BvM for mean functional (Bayesian bootstrap)",
    })


def cheatsheet():
    return "ghbvm: Bernstein-von Mises"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghbvm import ghosal_bernstein_von_mises
# >>> r = ghosal_bernstein_von_mises(np.random.default_rng(0).normal(size=200))
# >>> 0.0 <= r["z_ks_pvalue"] <= 1.0
# True
