"""A journey of a thousand miles begins with a single step. -- Lao Tzu"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st
from scipy.optimize import minimize

from ._containers import DescriptiveResult


def truncated_regression(y: np.ndarray, x: np.ndarray, threshold: float, direction: str = "left", cdf=None) -> DescriptiveResult:
    """
    Truncated regression via maximum likelihood.

    Models the outcome conditional on it exceeding (or falling below)
    *threshold*, accounting for the truncation in the likelihood.

    :param y: Observed outcome (1-D), only values beyond threshold.
    :param x: Regressors (n, p) or (n,).
    :param threshold: Truncation point.
    :param direction: 'left' (y > threshold) or 'right' (y < threshold).
    :return: DescriptiveResult with coefficient estimates.
    :raises ValueError: If inputs invalid or direction unrecognised.

    References
    ----------
    Greene, W. H. (2012). Econometric Analysis (7th ed.), Ch. 19. Pearson.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.shape[0] != x.shape[0] or y.size < 5:
        raise ValueError("y and x must have matching rows, with n >= 5.")
    if direction not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got '{direction}'.")

    n, p = x.shape
    X = np.column_stack([np.ones(n), x])
    k = X.shape[1]

    def neg_loglik(params):
        beta = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        mu = X @ beta
        z = (y - mu) / sigma
        z_c = (threshold - mu) / sigma
        ll = _st.norm.logpdf(z) - np.log(sigma)
        if direction == "left":
            ll -= np.log(np.maximum(1.0 - _st.norm.cdf(z_c), 1e-15))
        else:
            ll -= np.log(np.maximum(_st.norm.cdf(z_c), 1e-15))
        return -np.sum(ll)

    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma_init = np.log(np.std(y - X @ beta_init) + 1e-6)
    x0 = np.concatenate([beta_init, [sigma_init]])

    res = minimize(neg_loglik, x0, method="L-BFGS-B")
    beta_hat = res.x[:k]
    sigma_hat = float(np.exp(res.x[k]))

    return DescriptiveResult(
        name="Truncated Regression",
        value=float(beta_hat[1]) if k > 1 else float(beta_hat[0]),
        extra={
            "beta": beta_hat.tolist(),
            "sigma": sigma_hat,
            "threshold": threshold,
            "direction": direction,
            "converged": res.success,
            "log_likelihood": float(-res.fun),
            "n": n,
        },
    )


trunc = truncated_regression


def cheatsheet() -> str:
    return "truncated_regression({}) -> Truncated regression. 'All who gain power are afraid to lose"
