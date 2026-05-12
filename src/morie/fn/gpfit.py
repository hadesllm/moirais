# morie.fn -- function file (hadesllm/morie)
"""Generalised Pareto distribution fit (Pickands 1975); peaks-over-threshold.

ML-fits the two-parameter GP

    F(x) = 1 - (1 + xi x / sigma)^{-1/xi},   x > 0

to threshold exceedances ``x = y - u`` for chosen threshold ``u``.

Also exposes the legacy ``gpfit(coords=..., data=...)`` spatial Gaussian
process signature so existing morie.fn.Spatial callers
(``mcint``/``gpfit`` paired) keep working; the function dispatches on
which kwargs are supplied.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import SpatialResult
from ._richresult import RichResult

__all__ = ["generalized_pareto", "gpfit"]


def generalized_pareto(x, threshold: float | None = None):
    """Fit a Generalised Pareto distribution to threshold exceedances.

    Parameters
    ----------
    x : array-like
        Raw observations; only ``x > threshold`` are used.
    threshold : float, optional
        Threshold ``u``.  Default = 90th percentile of ``x``.

    Returns
    -------
    RichResult: scale (sigma), shape (xi), threshold, n_exceedances,
    se_sigma, se_xi, loglik, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 5:
        return RichResult(payload={"estimate": float("nan"), "n": int(x.size),
                                   "method": "GP (n<5)"})
    if threshold is None:
        threshold = float(np.quantile(x, 0.90))
    excess = x[x > threshold] - threshold
    n = excess.size
    if n < 5:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "GP (too few exceedances)"})
    # scipy genpareto: shape c = xi (matches Coles)
    c, loc, scale = stats.genpareto.fit(excess, floc=0)
    xi = float(c); sigma = float(scale)
    loglik = float(np.sum(stats.genpareto.logpdf(excess, c=c, loc=0, scale=scale)))
    # info matrix
    def nll(p):
        sig_, xi_ = p
        if sig_ <= 0:
            return np.inf
        return -np.sum(stats.genpareto.logpdf(excess, c=xi_, loc=0, scale=sig_))
    eps = 1e-4
    p0 = np.array([sigma, xi])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            pp = p0.copy(); pp[i] += eps; pp[j] += eps; f_pp = nll(pp)
            pm = p0.copy(); pm[i] += eps; pm[j] -= eps; f_pm = nll(pm)
            mp = p0.copy(); mp[i] -= eps; mp[j] += eps; f_mp = nll(mp)
            mm = p0.copy(); mm[i] -= eps; mm[j] -= eps; f_mm = nll(mm)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(H)
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        ses = np.full(2, np.nan)
    return RichResult(payload={
        "scale": sigma, "shape": xi, "threshold": float(threshold),
        "n_exceedances": int(n), "se_sigma": float(ses[0]),
        "se_xi": float(ses[1]), "loglik": loglik,
        "estimate": sigma, "se": float(ses[0]),
        "method": "GP MLE (Pickands 1975)",
    })


def gpfit(data=None, coords=None, x=None, threshold=None, n=100, seed=42,
          **kwargs):
    """Dispatcher: GP-Pareto fit when given a 1-D ``x``; otherwise spatial
    Gaussian-process placeholder (preserves legacy Spatial-suite shape).
    """
    if x is not None:
        return generalized_pareto(x, threshold=threshold)
    if data is not None and (coords is None
                             or np.asarray(coords).ndim == 1):
        return generalized_pareto(np.asarray(data, dtype=float),
                                  threshold=threshold)
    rng = np.random.default_rng(seed)
    if coords is None:
        coords = rng.uniform(0, 1, size=(n, 2))
    coords = np.asarray(coords, dtype=float)
    if data is not None:
        data = np.asarray(data, dtype=float)
        statistic = float(np.mean(data))
    else:
        statistic = float(rng.standard_normal())
    return SpatialResult(
        name="GP-Regression",
        statistic=statistic,
        p_value=None,
        extra={"n_points": int(coords.shape[0])},
    )


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.exponential(scale=1.0, size=1000)
# >>> res = generalized_pareto(x, threshold=0.5)
# >>> # exponential = GP with xi=0
# >>> assert abs(res["shape"]) < 0.2


def cheatsheet():
    return "gpfit(x, threshold=auto): Generalised Pareto MLE (POT)."
