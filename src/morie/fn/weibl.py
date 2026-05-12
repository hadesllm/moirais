"""Mastering others is strength; mastering yourself is true power. -- Lao Tzu"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_fn

from ._containers import DescriptiveResult


def weibull_fit(
    data: np.ndarray,
    censored: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Maximum likelihood estimation of Weibull distribution parameters.

    Fits the two-parameter Weibull with PDF:

    .. math::

        f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}
               \exp\left(-\left(\frac{t}{\lambda}\right)^k\right)

    Parameters
    ----------
    data : ndarray, shape (n,)
        Observed failure/event times (positive values).
    censored : ndarray or None, shape (n,)
        Boolean array where True = right-censored (did not fail).
        If None, all observations are treated as failures.

    Returns
    -------
    DescriptiveResult
        name='Weibull Fit', value=shape parameter (k),
        extra has 'shape', 'scale', 'mean_life', 'median_life',
        'log_likelihood'.

    References
    ----------
    Weibull, W. (1951). A statistical distribution function of wide
    applicability. *Journal of Applied Mechanics*, 18(3), 293-297.
    """
    t = np.asarray(data, dtype=np.float64).ravel()
    if np.any(t <= 0):
        raise ValueError("All failure times must be positive.")

    n = len(t)
    if censored is None:
        c = np.zeros(n, dtype=bool)
    else:
        c = np.asarray(censored, dtype=bool).ravel()

    d = (~c).astype(float)
    n_failures = d.sum()

    def neg_log_lik(k):
        if k <= 0:
            return 1e12
        log_t = np.log(t)
        tk = t**k
        lam_k = np.sum(tk) / max(n_failures, 1)
        lam = lam_k ** (1.0 / k)
        ll = n_failures * np.log(k) - n_failures * k * np.log(lam) + (k - 1) * np.sum(d * log_t) - np.sum(tk) / (lam**k)
        return -ll

    result = minimize_scalar(neg_log_lik, bounds=(0.1, 20.0), method="bounded")
    k = result.x

    lam = (np.sum(t**k) / max(n_failures, 1)) ** (1.0 / k)

    mean_life = lam * gamma_fn(1 + 1.0 / k)
    median_life = lam * (np.log(2)) ** (1.0 / k)
    ll = -result.fun

    return DescriptiveResult(
        name="Weibull Fit",
        value=float(k),
        extra={
            "shape": float(k),
            "scale": float(lam),
            "mean_life": float(mean_life),
            "median_life": float(median_life),
            "log_likelihood": float(ll),
            "n": n,
            "n_failures": int(n_failures),
        },
    )


weibl = weibull_fit


def cheatsheet() -> str:
    return "weibull_fit({}) -> Weibull distribution MLE fit. 'Power! Unlimited power!' -- D"
