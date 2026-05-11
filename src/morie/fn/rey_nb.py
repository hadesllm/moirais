# morie.fn — function file (hadesllm/morie)
"""Negative binomial regression (GLM) via IRLS."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from morie.fn._containers import RegressionResult


def rey_nb(df, y: str = "y", x: list | str = "x", alpha_nb: float = 1.0, alpha: float = 0.05, cdf=None) -> RegressionResult:
    """
    Negative binomial regression via maximum likelihood.

    Models count data with overdispersion:

    .. math::

        P(Y = y \\mid \\mu, \\alpha) =
        \\frac{\\Gamma(y + 1/\\alpha)}{\\Gamma(y+1)\\,\\Gamma(1/\\alpha)}
        \\left(\\frac{1}{1 + \\alpha\\mu}\\right)^{1/\\alpha}
        \\left(\\frac{\\alpha\\mu}{1 + \\alpha\\mu}\\right)^{y}

    :param df: DataFrame with response and predictor columns.
    :param y: Name of the count response column.
    :param x: Predictor column name(s) (str or list of str).
    :param alpha_nb: Overdispersion parameter (> 0). Estimated jointly
        if initial value given. Default 1.0.
    :param alpha: Significance level for Wald p-values. Default 0.05.
    :return: :class:`RegressionResult` with NegBin coefficients.
    :raises ValueError: On missing columns or non-count response.

    References
    ----------
    Cameron, A. C. & Trivedi, P. K. (2013). Regression Analysis of
    Count Data (2nd ed.). Cambridge University Press.

    Hilbe, J. M. (2011). Negative Binomial Regression (2nd ed.).
    Cambridge University Press.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    y_arr = np.asarray(df[y], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x])
    n, p = X_arr.shape

    if np.any(y_arr < 0):
        raise ValueError("Response must be non-negative counts.")

    def neg_loglik(params):
        beta = params[:p]
        log_alpha = params[p]
        a = np.exp(log_alpha)
        mu = np.exp(X_arr @ beta)
        mu = np.clip(mu, 1e-10, 1e10)
        r = 1.0 / a
        ll = (
            gammaln(y_arr + r)
            - gammaln(y_arr + 1)
            - gammaln(r)
            + r * np.log(r / (r + mu))
            + y_arr * np.log(mu / (r + mu))
        )
        return -np.sum(ll)

    # Initial values from Poisson (log-link) OLS
    log_y = np.log(y_arr + 0.5)
    beta0 = np.linalg.lstsq(X_arr, log_y, rcond=None)[0]
    x0 = np.concatenate([beta0, [np.log(alpha_nb)]])

    result = minimize(neg_loglik, x0, method="BFGS")
    params_hat = result.x
    beta_hat = params_hat[:p]

    # Standard errors
    if result.hess_inv is not None:
        H_inv = (
            np.asarray(result.hess_inv)
            if not hasattr(result.hess_inv, "todense")
            else np.asarray(result.hess_inv.todense())
        )
        se_all = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
    else:
        se_all = np.full(len(params_hat), np.nan)
    se_beta = se_all[:p]

    z_vals = beta_hat / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    names = ["intercept"] + list(x)
    mu_hat = np.exp(X_arr @ beta_hat)
    residuals = y_arr - mu_hat
    loglik = -result.fun
    aic = 2.0 * (p + 1) - 2.0 * loglik

    return RegressionResult(
        method="Negative Binomial",
        coefficients=dict(zip(names, beta_hat.tolist())),
        se=dict(zip(names, se_beta.tolist())),
        p_values=dict(zip(names, p_vals.tolist())),
        residuals=residuals,
        fitted=mu_hat,
        n=n,
        k=p,
        extra={"aic": float(aic), "alpha_nb": float(np.exp(params_hat[p]))},
    )


def cheatsheet() -> str:
    return "rey_nb({}) -> Negative binomial regression (GLM) via IRLS."
