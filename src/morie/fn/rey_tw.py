# morie.fn -- function file (hadesllm/morie)
"""Tweedie regression (compound Poisson-gamma GLM)."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from morie.fn._containers import RegressionResult


def rey_tw(df, y: str = "y", x: list | str = "x", power: float = 1.5, alpha: float = 0.05, cdf=None) -> RegressionResult:
    r"""
    Tweedie regression via quasi-likelihood maximisation.

    The Tweedie family unifies Poisson (*p* = 1), compound Poisson-gamma
    (1 < *p* < 2), and gamma (*p* = 2) distributions under a single
    variance function :math:`V(\\mu) = \\mu^p`.

    :param df: DataFrame with response and predictor columns.
    :param y: Response column name (non-negative).
    :param x: Predictor column name(s).
    :param power: Tweedie power parameter. Must be in (1, 2). Default 1.5.
    :param alpha: Significance level. Default 0.05.
    :return: :class:`RegressionResult` with Tweedie coefficients.
    :raises ValueError: On invalid power or negative response.

    References
    ----------
    Jorgensen, B. (1987). Exponential dispersion models. *Journal of
    the Royal Statistical Society: Series B*, 49(2), 127-162.

    Tweedie, M. C. K. (1984). An index which distinguishes between
    some important exponential families. In J. K. Ghosh & J. Roy (Eds.),
    *Statistics: Applications and New Directions*, pp. 579-604.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")
    if not (1.0 < power < 2.0):
        raise ValueError(f"power must be in (1, 2), got {power}.")

    y_arr = np.asarray(df[y], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x])
    n, p_dim = X_arr.shape

    if np.any(y_arr < 0):
        raise ValueError("Response must be non-negative for Tweedie regression.")

    pw = power

    def neg_quasi_ll(beta):
        eta = X_arr @ beta
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        # Tweedie deviance unit: 2 * [y*mu^(1-p)/(1-p) - mu^(2-p)/(2-p)]
        # quasi-loglik kernel
        if pw == 1:
            ll = np.sum(y_arr * np.log(mu) - mu)
        elif pw == 2:
            ll = np.sum(-y_arr / mu - np.log(mu))
        else:
            term1 = y_arr * mu ** (1 - pw) / (1 - pw)
            term2 = mu ** (2 - pw) / (2 - pw)
            ll = np.sum(term1 - term2)
        return -ll

    # Initialize with log-link OLS
    log_y = np.log(np.maximum(y_arr, 0.5))
    beta0 = np.linalg.lstsq(X_arr, log_y, rcond=None)[0]
    result = minimize(neg_quasi_ll, beta0, method="BFGS")
    beta_hat = result.x

    mu_hat = np.exp(X_arr @ beta_hat)
    mu_hat = np.clip(mu_hat, 1e-10, None)
    residuals = y_arr - mu_hat

    # Dispersion estimate
    V_mu = mu_hat**pw
    pearson_resid = (y_arr - mu_hat) / np.sqrt(V_mu)
    phi = np.sum(pearson_resid**2) / max(n - p_dim, 1)

    # Standard errors via Hessian
    if result.hess_inv is not None:
        H_inv = (
            np.asarray(result.hess_inv)
            if not hasattr(result.hess_inv, "todense")
            else np.asarray(result.hess_inv.todense())
        )
        se_beta = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
    else:
        se_beta = np.full(p_dim, np.nan)

    z_vals = beta_hat / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    # Tweedie deviance
    dev = 2.0 * np.sum(
        np.maximum(y_arr, 1e-10) ** (2 - pw) / ((1 - pw) * (2 - pw))
        - y_arr * mu_hat ** (1 - pw) / (1 - pw)
        + mu_hat ** (2 - pw) / (2 - pw)
    )
    aic = 2.0 * p_dim - 2.0 * (-result.fun)

    names = ["intercept"] + list(x)

    return RegressionResult(
        method="Tweedie GLM",
        coefficients=dict(zip(names, beta_hat.tolist())),
        se=dict(zip(names, se_beta.tolist())),
        p_values=dict(zip(names, p_vals.tolist())),
        residuals=residuals,
        fitted=mu_hat,
        n=n,
        k=p_dim,
        extra={"aic": float(aic), "power": power, "phi": float(phi), "deviance": float(dev)},
    )


def cheatsheet() -> str:
    return "rey_tw({}) -> Tweedie regression (compound Poisson-gamma GLM)."
