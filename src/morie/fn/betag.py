# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Beta regression for rates/proportions in (0,1)."""

from __future__ import annotations

import numpy as np
from scipy import optimize, special
from scipy import stats as _st

from ._containers import RegressionResult


def beta_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
) -> RegressionResult:
    r"""Beta regression via MLE (logit link).

    Models a response in (0, 1) with a Beta distribution parameterised
    by mean :math:`\\mu = \\text{logit}^{-1}(X\\beta)` and precision
    :math:`\\phi`.

    Parameters
    ----------
    y : (n,) responses strictly in (0, 1)
    X : (n, p) predictors
    add_intercept : bool

    Returns
    -------
    RegressionResult

    References
    ----------
    Ferrari, S. L. P. & Cribari-Neto, F. (2004). Beta regression for
    modelling rates and proportions. *J. Appl. Stat.*, 31(7), 799--815.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.any(y <= 0) or np.any(y >= 1):
        raise ValueError("Beta regression requires y in (0, 1).")
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    def neg_loglik(params):
        beta = params[:k]
        log_phi = params[k]
        phi = np.exp(log_phi)
        mu = special.expit(X @ beta)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        a = mu * phi
        b = (1 - mu) * phi
        ll = np.sum(
            special.gammaln(phi)
            - special.gammaln(a)
            - special.gammaln(b)
            + (a - 1) * np.log(y)
            + (b - 1) * np.log(1 - y)
        )
        return -ll

    beta0 = np.zeros(k)
    beta0[0] = np.log(np.mean(y) / (1 - np.mean(y)))
    phi0 = np.log(10.0)
    x0 = np.append(beta0, phi0)

    res = optimize.minimize(neg_loglik, x0, method="BFGS")
    beta = res.x[:k]
    phi = np.exp(res.x[k])

    mu_f = special.expit(X @ beta)
    mu_f = np.clip(mu_f, 1e-8, 1 - 1e-8)

    try:
        H = res.hess_inv if hasattr(res, "hess_inv") and res.hess_inv is not None else np.eye(k + 1)
        if isinstance(H, np.ndarray):
            se_all = np.sqrt(np.diag(H).clip(0))
        else:
            se_all = np.full(k + 1, float("nan"))
        se_arr = se_all[:k]
    except Exception:
        se_arr = np.full(k, float("nan"))

    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="Beta Regression",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=y - mu_f,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"phi": float(phi), "log_likelihood": float(-res.fun)},
    )


betag = beta_regression


def cheatsheet() -> str:
    return "beta_regression({}) -> Beta regression for rates/proportions in (0,1)."
