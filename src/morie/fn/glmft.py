# morie.fn — function file (hadesllm/morie)
"""Generalized linear model fitting."""

from __future__ import annotations

import numpy as np

from ._containers import RegressionResult

_FAMILIES = {
    "gaussian": "Gaussian",
    "binomial": "Binomial",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "inverse_gaussian": "InverseGaussian",
    "negative_binomial": "NegativeBinomial",
}

_DEFAULT_LINKS = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
    "inverse_gaussian": "inverse_squared",
    "negative_binomial": "log",
}


def glm_fit(y: np.ndarray, X: np.ndarray, cdf=None, *, family: str = "gaussian", link: str | None = None, max_iter: int = 50, tol: float = 1e-8) -> RegressionResult:
    """Generalized linear model via IRLS.

    :param y: Response variable (n,).
    :param X: Predictor matrix (n, p). Intercept added automatically.
    :param family: One of gaussian, binomial, poisson, gamma,
        inverse_gaussian, negative_binomial.
    :param link: Link function. Auto-selected from family if None.
    :param max_iter: Maximum IRLS iterations.
    :param tol: Convergence tolerance.
    :return: RegressionResult with coefficients, SE, p-values, deviance, AIC.
    :raises ValueError: If family is not recognized.
    """
    from scipy import stats as sp_stats

    family = family.lower()
    if family not in _FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from: {list(_FAMILIES)}")
    if link is None:
        link = _DEFAULT_LINKS[family]

    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    def _link_fn(mu):
        if link == "identity":
            return mu
        if link == "log":
            return np.log(mu + 1e-12)
        if link == "logit":
            return np.log((mu + 1e-12) / (1 - mu + 1e-12))
        if link == "inverse":
            return 1.0 / (mu + 1e-12)
        if link == "inverse_squared":
            return 1.0 / (mu**2 + 1e-12)
        if link == "probit":
            return sp_stats.norm.ppf(np.clip(mu, 1e-8, 1 - 1e-8))
        if link == "cloglog":
            return np.log(-np.log(1 - np.clip(mu, 1e-8, 1 - 1e-8)))
        if link == "sqrt":
            return np.sqrt(mu + 1e-12)
        return mu

    def _inv_link(eta):
        if link == "identity":
            return eta
        if link == "log":
            return np.exp(np.clip(eta, -20, 20))
        if link == "logit":
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
        if link == "inverse":
            return 1.0 / (eta + 1e-12)
        if link == "inverse_squared":
            return 1.0 / np.sqrt(np.abs(eta) + 1e-12)
        if link == "probit":
            return sp_stats.norm.cdf(eta)
        if link == "cloglog":
            return 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))
        if link == "sqrt":
            return eta**2
        return eta

    def _dmu_deta(eta):
        if link == "identity":
            return np.ones_like(eta)
        if link == "log":
            return np.exp(np.clip(eta, -20, 20))
        if link == "logit":
            mu = _inv_link(eta)
            return mu * (1 - mu) + 1e-12
        if link == "inverse":
            return -1.0 / (eta**2 + 1e-12)
        if link == "sqrt":
            return 2.0 * eta
        return np.ones_like(eta)

    def _variance(mu):
        if family == "gaussian":
            return np.ones_like(mu)
        if family == "binomial":
            return mu * (1 - mu) + 1e-12
        if family == "poisson":
            return mu + 1e-12
        if family == "gamma":
            return mu**2 + 1e-12
        if family == "inverse_gaussian":
            return mu**3 + 1e-12
        if family == "negative_binomial":
            return mu + mu**2 + 1e-12
        return np.ones_like(mu)

    if family == "gaussian":
        mu_init = y.copy()
    elif family == "binomial":
        mu_init = (y + 0.5) / 2.0
    else:
        mu_init = np.where(y > 0, y, 0.1)

    eta = _link_fn(mu_init)
    beta = np.zeros(k)

    for _ in range(max_iter):
        mu = _inv_link(eta)
        dmu = _dmu_deta(eta)
        v = _variance(mu)
        w = dmu**2 / (v + 1e-12)
        z = eta + (y - mu) / (dmu + 1e-12)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X_int.T @ W @ X_int, X_int.T @ W @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
        eta = X_int @ beta

    mu_final = _inv_link(X_int @ beta)

    if family == "gaussian":
        dev = float(np.sum((y - mu_final) ** 2))
    elif family == "binomial":
        dev = float(
            2
            * np.sum(
                y * np.log((y + 1e-12) / (mu_final + 1e-12))
                + (1 - y) * np.log((1 - y + 1e-12) / (1 - mu_final + 1e-12))
            )
        )
    elif family == "poisson":
        dev = float(2 * np.sum(y * np.log((y + 1e-12) / (mu_final + 1e-12)) - (y - mu_final)))
    elif family == "gamma":
        dev = float(2 * np.sum(-np.log((y + 1e-12) / (mu_final + 1e-12)) + (y - mu_final) / (mu_final + 1e-12)))
    else:
        dev = float(np.sum((y - mu_final) ** 2 / (_variance(mu_final) + 1e-12)))

    aic = dev + 2 * k

    try:
        v_final = _variance(mu_final)
        dmu_final = _dmu_deta(X_int @ beta)
        w_final = dmu_final**2 / (v_final + 1e-12)
        cov_beta = np.linalg.inv(X_int.T @ np.diag(w_final) @ X_int)
        se = np.sqrt(np.abs(np.diag(cov_beta)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    z_vals = beta / (se + 1e-12)
    p_vals = 2 * sp_stats.norm.sf(np.abs(z_vals))

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    mu_null = np.full(n, np.mean(y))
    if family == "gaussian":
        null_dev = float(np.sum((y - mu_null) ** 2))
    elif family == "poisson":
        null_dev = float(2 * np.sum(y * np.log((y + 1e-12) / (mu_null + 1e-12)) - (y - mu_null)))
    else:
        null_dev = dev

    pseudo_r2 = 1.0 - dev / null_dev if null_dev > 0 else None

    return RegressionResult(
        method=f"GLM({_FAMILIES[family]}/{link})",
        coefficients=dict(zip(coef_names, beta.tolist())),
        se=dict(zip(coef_names, se.tolist())),
        p_values=dict(zip(coef_names, p_vals.tolist())),
        r_squared=pseudo_r2 if family != "gaussian" else float(1 - dev / null_dev) if null_dev > 0 else None,
        n=n,
        k=k,
        extra={
            "family": family,
            "link": link,
            "deviance": dev,
            "null_deviance": null_dev,
            "aic": aic,
        },
    )


glmft = glm_fit


def cheatsheet() -> str:
    return "glm_fit({}) -> Generalized linear model fitting."
