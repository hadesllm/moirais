# morie.fn — function file (hadesllm/morie)
"""Beta-process prior for survival (Hjort 1990)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_survival_beta_process", "ghsrv"]


def ghosal_survival_beta_process(time, event=None, c=1.0, lam0=None):
    """Posterior mean survival under a beta-process prior on the
    cumulative-hazard ``H``.

    Hjort (1990) showed that for right-censored survival data with
    prior ``H ~ BP(c, H_0)`` the posterior is again a beta process
    with parameters

        c_post(t)  = c(t) + Y(t-),
        dH_post(t) = (c(t) dH_0(t) + dN(t)) / (c(t) + Y(t-))

    where ``N(t)`` is the counting process of events and ``Y(t)`` the
    at-risk process.  The posterior-mean survival function is

        S_hat(t) = prod_{s <= t} (1 - dH_post(s)).

    When ``c`` is large, ``S_hat`` shrinks toward ``S_0 = exp(-H_0)``;
    when ``c -> 0`` it converges to the Kaplan–Meier estimator.

    Parameters
    ----------
    time : array-like — observation times (possibly censored).
    event : array-like or None — 1=event, 0=censored.  All-events if None.
    c : float — prior concentration (constant in t).
    lam0 : float or None — exponential base hazard rate.  Defaults to
        ``1/mean(time)``.

    Returns
    -------
    RichResult with ``estimate`` (S_hat at median time), ``S_post``,
    ``times``, ``H_post`` (cumulative hazard at event times).

    References
    ----------
    Hjort, N. L. (1990). Nonparametric Bayes estimators based on
      beta processes in models for life-history data. AOS 18.
    Ghosal & van der Vaart (2017) Ch 14.
    """
    t = np.asarray(time, dtype=float).ravel()
    n = int(t.size)
    if n == 0:
        return RichResult(payload={
            "estimate": float("nan"), "n": 0,
            "method": "Beta-process survival (empty input)",
        })
    if event is None:
        d = np.ones(n, dtype=int)
    else:
        d = np.asarray(event, dtype=int).ravel()
    if lam0 is None:
        lam0 = 1.0 / max(float(np.mean(t)), 1e-6)
    order = np.argsort(t)
    t_s = t[order]
    d_s = d[order]
    # Unique event times
    uniq, inv = np.unique(t_s, return_inverse=True)
    Y = np.empty_like(uniq, dtype=float)
    dN = np.zeros_like(uniq, dtype=float)
    for k, tk in enumerate(uniq):
        Y[k] = float((t_s >= tk).sum())
        dN[k] = float(((t_s == tk) & (d_s == 1)).sum())
    # Base hazard increment dH0 = lam0 * (t_k - t_{k-1})
    dH0 = np.diff(np.concatenate(([0.0], uniq))) * lam0
    dH_post = (c * dH0 + dN) / (c + Y)
    S_post = np.cumprod(1.0 - np.clip(dH_post, 0, 1 - 1e-12))
    H_post = np.cumsum(dH_post)
    t_med = float(np.median(t))
    idx = int(np.searchsorted(uniq, t_med, side="right") - 1)
    S_at_med = float(S_post[idx]) if 0 <= idx < len(S_post) else 1.0
    return RichResult(payload={
        "estimate": S_at_med,
        "times": uniq.tolist(),
        "S_post": S_post.tolist(),
        "H_post": H_post.tolist(),
        "c": float(c),
        "lam0": float(lam0),
        "n": n,
        "method": "Beta-process posterior survival (Hjort 1990)",
    })


# Back-compat alias for the legacy `ghsrv` import in fn/__init__.py.
ghsrv = ghosal_survival_beta_process


def cheatsheet():
    return "ghsrv: Beta-process prior survival"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghsrv import ghosal_survival_beta_process
# >>> rng = np.random.default_rng(0)
# >>> t = rng.exponential(scale=2.0, size=80)
# >>> ev = rng.binomial(1, 0.8, size=80)
# >>> r = ghosal_survival_beta_process(t, ev)
# >>> 0.0 <= r["estimate"] <= 1.0
# True
