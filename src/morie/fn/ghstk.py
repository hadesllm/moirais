# morie.fn — function file (hadesllm/morie)
"""Truncated stick-breaking representation of a DP."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_stick_breaking_trunc"]


def ghosal_stick_breaking_trunc(x, alpha=1.0, K=50, seed=0,
                                  base_mean=None, base_sd=None,
                                  deterministic_seed: int | None = None):
    """Draw a truncated stick-breaking representation of ``DP(alpha, G0)``.

    Sethuraman (1994) representation::

        V_k ~ Beta(1, alpha),  w_k = V_k prod_{j<k}(1 - V_j),
        theta_k ~ G0,
        G_K = sum_{k=1}^K w_k delta_{theta_k}.

    The truncation error in total variation is bounded by
    ``E[1 - sum_{k=1}^K w_k] = (alpha/(alpha+1))^K``
    (Ishwaran & James 2001).

    The base measure ``G0`` defaults to ``N(mean(x), sd(x)^2)``.

    Parameters
    ----------
    x : array-like
        Used to set the base measure if ``base_mean``/``base_sd``
        are omitted, and to compute the headline estimate as
        ``G_K(t)`` at ``t = mean(x)``.
    alpha : float
        Concentration.
    K : int
        Truncation level.
    seed : int
    base_mean, base_sd : float, optional
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with keys ``estimate`` (truncated CDF at sample mean),
    ``weights``, ``atoms``, ``effective_K``, ``trunc_err_bound``.

    References
    ----------
    Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
      Statistica Sinica 4.
    Ishwaran, H. & James, L. (2001). Gibbs Sampling Methods for
      Stick-Breaking Priors. JASA 96.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ghstk", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if base_mean is None:
        base_mean = float(np.mean(x)) if n else 0.0
    if base_sd is None:
        base_sd = float(np.std(x, ddof=1)) if n > 1 else 1.0
        base_sd = max(base_sd, 1e-6)
    V = rng.beta(1.0, alpha, size=K)
    log1mV = np.log1p(-V)
    log_cumprod = np.concatenate(([0.0], np.cumsum(log1mV[:-1])))
    w = V * np.exp(log_cumprod)
    theta = rng.normal(loc=base_mean, scale=base_sd, size=K)
    t0 = float(np.mean(x)) if n else base_mean
    # Headline: G_K((-inf, t0]) = sum_k w_k * 1{theta_k <= t0}.
    estimate = float(np.sum(w * (theta <= t0)))
    trunc_bound = (alpha / (alpha + 1.0)) ** K
    return RichResult(payload={
        "estimate": estimate,
        "weights": w.tolist(),
        "atoms": theta.tolist(),
        "effective_K": int(np.sum(w > 1e-3)),
        "trunc_err_bound": float(trunc_bound),
        "n": n,
        "method": "Truncated stick-breaking DP draw (Sethuraman 1994)",
    })


def cheatsheet():
    return "ghstk: Truncated stick-breaking representation"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghstk import ghosal_stick_breaking_trunc
# >>> r = ghosal_stick_breaking_trunc(np.zeros(10), alpha=2.0, K=200, seed=0)
# >>> abs(sum(r["weights"]) - 1) < 1e-3
# True
