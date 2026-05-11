# morie.fn — function file (hadesllm/morie)
"""Gamma GLM regression via IRLS."""

import numpy as np
from scipy.stats import norm

from morie.fn._containers import RegressionResult


def rey_gm(df, y: str = "y", x: list | str = "x", link: str = "log", max_iter: int = 50, tol: float = 1e-8, cdf=None) -> RegressionResult:
    """
    Gamma GLM regression via iteratively reweighted least squares.

    Uses the log link by default:

    .. math::

        \\log E[Y \\mid X] = X\\beta, \\quad Y \\sim \\text{Gamma}(\\nu, \\mu)

    :param df: DataFrame with response and predictor columns.
    :param y: Response column name (must be positive).
    :param x: Predictor column name(s).
    :param link: Link function. ``"log"`` (default) or ``"inverse"``.
    :param max_iter: Maximum IRLS iterations. Default 50.
    :param tol: Convergence tolerance. Default 1e-8.
    :return: :class:`RegressionResult` with Gamma GLM coefficients.
    :raises ValueError: On non-positive response or unknown link.

    References
    ----------
    McCullagh, P. & Nelder, J. A. (1989). Generalized Linear Models
    (2nd ed.). Chapman & Hall.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    y_arr = np.asarray(df[y], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x])
    n, p = X_arr.shape

    if np.any(y_arr <= 0):
        raise ValueError("Response must be positive for Gamma regression.")
    if link not in ("log", "inverse"):
        raise ValueError(f"link must be 'log' or 'inverse', got {link!r}.")

    # Link functions
    if link == "log":
        g = np.log
        g_inv = np.exp
        g_deriv = lambda mu: 1.0 / mu
    else:  # inverse
        g = lambda mu: 1.0 / mu
        g_inv = lambda eta: 1.0 / eta
        g_deriv = lambda mu: -1.0 / (mu**2)

    # Initialize
    mu = y_arr.copy()
    mu = np.clip(mu, 1e-10, None)
    eta = g(mu)
    beta = np.linalg.lstsq(X_arr, eta, rcond=None)[0]

    for iteration in range(max_iter):
        eta = X_arr @ beta
        mu = g_inv(eta)
        mu = np.clip(mu, 1e-10, None)

        # IRLS weights for Gamma: V(mu) = mu^2
        d_eta = g_deriv(mu)
        V_mu = mu**2
        W = 1.0 / (d_eta**2 * V_mu)
        W = np.clip(W, 1e-10, 1e10)

        # Working response
        z = eta + (y_arr - mu) * d_eta

        # Weighted least squares
        Wdiag = np.diag(W)
        XtWX = X_arr.T @ Wdiag @ X_arr
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        beta_new = XtWX_inv @ X_arr.T @ Wdiag @ z

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final values
    eta = X_arr @ beta
    mu = g_inv(eta)
    mu = np.clip(mu, 1e-10, None)
    residuals = y_arr - mu

    # Dispersion (phi) estimate
    pearson_resid = (y_arr - mu) / mu
    phi = np.sum(pearson_resid**2) / max(n - p, 1)

    # Standard errors
    d_eta = g_deriv(mu)
    V_mu = mu**2
    W = 1.0 / (d_eta**2 * V_mu)
    W = np.clip(W, 1e-10, 1e10)
    XtWX = X_arr.T @ np.diag(W) @ X_arr
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
    se_beta = np.sqrt(np.maximum(np.diag(XtWX_inv) * phi, 0.0))

    z_vals = beta / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    # Deviance
    dev = 2.0 * np.sum((y_arr - mu) / mu - np.log(y_arr / mu))
    loglik = -0.5 * n * (1 + np.log(2 * np.pi * phi)) - dev / (2 * phi)
    aic = 2.0 * (p + 1) - 2.0 * loglik

    names = ["intercept"] + list(x)

    return RegressionResult(
        method="Gamma GLM",
        coefficients=dict(zip(names, beta.tolist())),
        se=dict(zip(names, se_beta.tolist())),
        p_values=dict(zip(names, p_vals.tolist())),
        residuals=residuals,
        fitted=mu,
        n=n,
        k=p,
        extra={"aic": float(aic), "deviance": float(dev), "phi": float(phi), "link": link},
    )


def cheatsheet() -> str:
    return "rey_gm({}) -> Gamma GLM regression via IRLS."
