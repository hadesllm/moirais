# morie.fn — function file (hadesllm/morie)
"""Accelerated failure time (AFT) model. 'I have lived most of my life surrounded by enemies.' -- Gamora"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ._containers import SurvivalResult


def aft_model(
    time: np.ndarray | list[float],
    event: np.ndarray | list[int],
    X: np.ndarray | None = None,
    *,
    distribution: str = "lognormal",
) -> SurvivalResult:
    r"""Fit an accelerated failure time (AFT) survival model.

    Parametric survival model where covariates accelerate or decelerate
    time-to-event: :math:`\\ln T = X\\beta + \\sigma W`.

    Parameters
    ----------
    time : array-like
        Observed times (positive).
    event : array-like
        Event indicator (1 = event, 0 = censored).
    X : np.ndarray or None
        Covariate matrix (n x p). If None, intercept-only model.
    distribution : str
        Error distribution: "lognormal" or "loglogistic".

    Returns
    -------
    SurvivalResult
        With ``extra`` containing ``coefficients``, ``se``, ``sigma``,
        ``log_likelihood``, ``aic``.
    """
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=int)
    if len(t) != len(d):
        raise ValueError("time and event must have same length")
    if np.any(t <= 0):
        raise ValueError("All times must be positive")

    n = len(t)
    y = np.log(t)

    if X is None:
        X_aug = np.ones((n, 1))
    else:
        X_aug = np.column_stack([np.ones(n), np.asarray(X, dtype=float)])
    p = X_aug.shape[1]

    if distribution == "lognormal":

        def _negloglik(params):
            beta = params[:p]
            log_sigma = params[p]
            sigma = np.exp(log_sigma)
            z = (y - X_aug @ beta) / sigma
            ll = np.sum(d * (stats.norm.logpdf(z) - np.log(sigma))) + np.sum((1 - d) * stats.norm.logsf(z))
            return -ll
    elif distribution == "loglogistic":

        def _negloglik(params):
            beta = params[:p]
            log_sigma = params[p]
            sigma = np.exp(log_sigma)
            z = (y - X_aug @ beta) / sigma
            ll = np.sum(d * (stats.logistic.logpdf(z) - np.log(sigma))) + np.sum((1 - d) * stats.logistic.logsf(z))
            return -ll
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    x0 = np.zeros(p + 1)
    x0[p] = np.log(np.std(y) + 1e-6)
    res = minimize(_negloglik, x0, method="BFGS")

    beta_hat = res.x[:p]
    sigma_hat = np.exp(res.x[p])
    ll = -res.fun

    try:
        hess_inv = res.hess_inv if hasattr(res, "hess_inv") else np.eye(p + 1)
        se = np.sqrt(np.abs(np.diag(hess_inv)))
    except Exception:
        se = np.full(p + 1, np.nan)

    aic = 2 * (p + 1) - 2 * ll

    return SurvivalResult(
        name=f"AFT ({distribution})",
        times=t,
        n_events=int(d.sum()),
        n_censored=int((1 - d).sum()),
        extra={
            "coefficients": dict(enumerate(beta_hat.tolist())),
            "se": se[:p].tolist(),
            "sigma": float(sigma_hat),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "distribution": distribution,
        },
    )


gamra = aft_model


def cheatsheet() -> str:
    return "aft_model({}) -> Accelerated failure time (AFT) model. 'I have lived most of "
