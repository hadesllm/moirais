# morie.fn — function file (hadesllm/morie)
"""Neutral-to-the-right process — Doksum (1974) survival prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_neutral_right"]


def ghosal_neutral_right(time, event=None, c=1.0, lam0=None):
    """Posterior survival under a neutral-to-the-right (NTR) process prior.

    A random distribution ``F`` is neutral-to-the-right (Doksum 1974) if
    ``F(t) = 1 - exp(-Y(t))`` for an independent-increment process
    ``Y(t)``.  Beta processes (Hjort 1990) and Dirichlet processes are
    both special cases.  We use the Doksum representation with a
    Gamma-process ``Y``: ``Y(t) - Y(s) ~ Gamma(c(t-s), 1)``, which gives
    posterior cumulative-hazard increments

        dH_post(t) = (c dH_0(t) + dN(t)) / (c + Y(t-)),
        S_post(t)  = prod_{s <= t} (1 - dH_post(s)).

    Identical in form to the beta-process posterior, but interpreted
    via the NTR representation.  The headline statistic is the
    posterior-mean ``S(t_med)``.

    Parameters
    ----------
    time : array-like.
    event : array-like or None.
    c : float (NTR concentration).
    lam0 : float or None.

    Returns
    -------
    RichResult with ``estimate`` (S at median time), ``S_post``,
    ``times``, ``H_post``, ``c``, ``lam0``.

    References
    ----------
    Doksum, K. A. (1974). Tailfree and neutral random probabilities.
      AOP 2.
    Ghosal & van der Vaart (2017) Ch 14.
    """
    t = np.asarray(time, dtype=float).ravel()
    n = int(t.size)
    if n == 0:
        return RichResult(payload={
            "estimate": float("nan"), "n": 0,
            "method": "NTR process (empty input)",
        })
    if event is None:
        d = np.ones(n, dtype=int)
    else:
        d = np.asarray(event, dtype=int).ravel()
    if lam0 is None:
        lam0 = 1.0 / max(float(np.mean(t)), 1e-6)
    t_s = np.sort(t)
    d_s = d[np.argsort(t)]
    uniq = np.unique(t_s)
    Y = np.array([(t_s >= tk).sum() for tk in uniq], dtype=float)
    dN = np.array([((t_s == tk) & (d_s == 1)).sum() for tk in uniq], dtype=float)
    dH0 = np.diff(np.concatenate(([0.0], uniq))) * lam0
    dH_post = (c * dH0 + dN) / (c + Y)
    S_post = np.cumprod(1.0 - np.clip(dH_post, 0, 1 - 1e-12))
    t_med = float(np.median(t))
    idx = int(np.searchsorted(uniq, t_med, side="right") - 1)
    S_at_med = float(S_post[idx]) if 0 <= idx < len(S_post) else 1.0
    return RichResult(payload={
        "estimate": S_at_med,
        "times": uniq.tolist(),
        "S_post": S_post.tolist(),
        "H_post": np.cumsum(dH_post).tolist(),
        "c": float(c),
        "lam0": float(lam0),
        "n": n,
        "method": "Neutral-to-the-right posterior (Doksum 1974)",
    })


def cheatsheet():
    return "ghntr: Neutral-to-the-right process"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghntr import ghosal_neutral_right
# >>> rng = np.random.default_rng(0)
# >>> r = ghosal_neutral_right(rng.exponential(size=60))
# >>> 0 <= r["estimate"] <= 1
# True
