# morie.fn -- function file (hadesllm/morie)
"""Censored (Tobit) regression. 'Much to learn, you still have.'"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st
from scipy.optimize import minimize

from ._containers import DescriptiveResult


def censored_regression(y: np.ndarray, x: np.ndarray, censor_val: float = 0.0, cdf=None) -> DescriptiveResult:
    r"""
    Tobit (Type I) censored regression via maximum likelihood.

    Left-censored at *censor_val*: observed :math:`y^* = \\max(y, c)`.

    :param y: Observed (possibly censored) outcome (1-D).
    :param x: Regressors (n, p) or (n,).
    :param censor_val: Censoring threshold. Default 0.0.
    :return: DescriptiveResult with coefficient estimates.
    :raises ValueError: If inputs invalid.

    References
    ----------
    Tobin, J. (1958). Estimation of relationships for limited dependent
    variables. Econometrica, 26(1), 24--36. doi:10.2307/1907382
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.shape[0] != x.shape[0] or y.size < 5:
        raise ValueError("y and x must have matching rows, with n >= 5.")

    n, p = x.shape
    X = np.column_stack([np.ones(n), x])
    k = X.shape[1]
    censored = y <= censor_val

    def neg_loglik(params):
        beta = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        mu = X @ beta
        z = (y - mu) / sigma

        ll = np.zeros(n)
        ll[~censored] = _st.norm.logpdf(z[~censored]) - np.log(sigma)
        z_c = (censor_val - mu[censored]) / sigma
        ll[censored] = np.log(np.maximum(_st.norm.cdf(z_c), 1e-15))
        return -np.sum(ll)

    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma_init = np.log(np.std(y - X @ beta_init) + 1e-6)
    x0 = np.concatenate([beta_init, [sigma_init]])

    res = minimize(neg_loglik, x0, method="L-BFGS-B")
    beta_hat = res.x[:k]
    sigma_hat = float(np.exp(res.x[k]))

    return DescriptiveResult(
        name="Tobit Regression",
        value=float(beta_hat[1]) if k > 1 else float(beta_hat[0]),
        extra={
            "beta": beta_hat.tolist(),
            "sigma": sigma_hat,
            "censor_val": censor_val,
            "n_censored": int(np.sum(censored)),
            "n_uncensored": int(np.sum(~censored)),
            "converged": res.success,
            "log_likelihood": float(-res.fun),
            "n": n,
        },
    )


censr = censored_regression


def cheatsheet() -> str:
    return "censored_regression({}) -> Censored (Tobit) regression. 'Much to learn, you still have."
