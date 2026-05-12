# morie.fn -- function file (hadesllm/morie)
"""Posterior contraction rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_contraction_rate"]


def ghosal_contraction_rate(x, beta=1.0, d=1):
    """Minimax posterior-contraction rate for Hölder-β densities.

    For an i.i.d. sample of size ``n`` from a Hölder-β density on R^d,
    the minimax posterior-contraction rate (Ghosal–Ghosh–van der Vaart
    2000, Ghosal Ch 6) is

        eps_n = n^{-beta / (2 beta + d)}.

    Up to log factors this rate is attained by Bayesian density
    estimators built on log-spline / wavelet / Gaussian sieves.  The
    function also reports the canonical parametric rate ``n^{-1/2}``
    for comparison.

    Parameters
    ----------
    x : array-like
        Sample (only its length matters).
    beta : float
        Smoothness index.
    d : int
        Dimension.

    Returns
    -------
    RichResult with ``estimate`` (eps_n), ``n``, ``beta``, ``d``,
    ``parametric_rate``, ``log_rate_correction``.

    References
    ----------
    Ghosal, Ghosh, van der Vaart (2000). AOS 28(2).
    Ghosal & van der Vaart (2017) Ch 6.
    """
    x = np.asarray(x).ravel()
    n = int(x.size)
    if n <= 1:
        return RichResult(payload={
            "estimate": float("nan"), "n": n,
            "method": "Contraction rate (n too small)",
        })
    eps_n = float(n ** (-beta / (2.0 * beta + d)))
    log_corr = float((np.log(n)) ** (beta / (2.0 * beta + d))) * eps_n
    return RichResult(payload={
        "estimate": eps_n,
        "log_rate_correction": log_corr,
        "parametric_rate": float(n ** -0.5),
        "n": n,
        "beta": float(beta),
        "d": int(d),
        "method": "Minimax contraction rate n^{-beta/(2beta+d)}",
    })


def cheatsheet():
    return "ghcrt: Posterior contraction rate"


# CANONICAL TEST
# >>> from morie.fn.ghcrt import ghosal_contraction_rate
# >>> r = ghosal_contraction_rate(list(range(100)), beta=1.0, d=1)
# >>> abs(r.estimate - 100**(-1/3)) < 1e-9
# True
