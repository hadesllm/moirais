# morie.fn -- function file (hadesllm/morie)
"""Adaptive posterior contraction over a range of smoothnesses."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_adaptation"]


def ghosal_adaptation(x, betas=None, d=1):
    """Adaptive contraction rates over a grid of Hölder smoothnesses.

    A hierarchical prior on smoothness (van der Vaart & van Zanten 2009,
    Ghosal Ch 10) yields the data-driven (adaptive) rate

        eps_n(beta) = n^{-beta / (2 beta + d)}  (up to log factors).

    Without knowing β, an adaptive Bayesian estimator achieves this
    rate uniformly over a class of β values.  This callable returns
    the full eps_n(β) curve and the smallest (best) rate.

    Parameters
    ----------
    x : array-like (used only for n).
    betas : iterable of float, default np.linspace(0.5, 3, 11).
    d : int.

    Returns
    -------
    RichResult with ``estimate`` (minimum rate over the grid),
    ``betas``, ``rates``, ``best_beta``.

    References
    ----------
    van der Vaart & van Zanten (2009). AOS 37.
    Ghosal & van der Vaart (2017) Ch 10.
    """
    x = np.asarray(x).ravel()
    n = int(x.size)
    if betas is None:
        betas = np.linspace(0.5, 3.0, 11)
    betas = np.asarray(betas, dtype=float)
    rates = n ** (-betas / (2.0 * betas + d))
    best = int(np.argmin(rates))
    return RichResult(payload={
        "estimate": float(rates[best]),
        "betas": betas.tolist(),
        "rates": rates.tolist(),
        "best_beta": float(betas[best]),
        "n": n,
        "d": int(d),
        "method": "Adaptive posterior contraction over Holder grid",
    })


def cheatsheet():
    return "ghadp: Adaptive posterior contraction"


# CANONICAL TEST
# >>> from morie.fn.ghadp import ghosal_adaptation
# >>> r = ghosal_adaptation(list(range(100)))
# >>> r["best_beta"] >= 0.5
# True
