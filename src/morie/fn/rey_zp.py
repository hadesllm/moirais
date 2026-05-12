# morie.fn -- function file (hadesllm/morie)
"""Zero-inflated Poisson regression via EM algorithm."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def rey_zp(df, y: str = "y", x: list | str = "x", max_iter: int = 200, tol: float = 1e-6, cdf=None) -> dict:
    r"""
    Zero-inflated Poisson (ZIP) regression via EM algorithm.

    Two-component mixture: a point mass at zero (logistic model) and a
    Poisson count model:

    .. math::

        P(Y = 0) = \\pi + (1 - \\pi) e^{-\\lambda}

        P(Y = y) = (1 - \\pi) \\frac{\\lambda^y e^{-\\lambda}}{y!},
        \\quad y = 1, 2, \\ldots

    :param df: DataFrame with response and predictor columns.
    :param y: Count response column name (non-negative integers).
    :param x: Predictor column name(s).
    :param max_iter: Maximum EM iterations. Default 200.
    :param tol: Convergence tolerance. Default 1e-6.
    :return: dict with ``zero_coefficients`` (logistic part),
        ``count_coefficients`` (Poisson part), ``zero_se``, ``count_se``,
        ``zero_p_values``, ``count_p_values``, ``aic``, ``n``.
    :raises ValueError: On missing columns or negative counts.

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression with an
    application to defects in manufacturing. *Technometrics*, 34(1), 1-14.
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

    # Indicator for zero
    is_zero = (y_arr == 0).astype(float)

    # Initialize pi (zero-inflation probability) and lambda
    pi_hat = np.mean(is_zero) * 0.5
    # Poisson MLE on non-zero as start
    nz = y_arr > 0
    if np.sum(nz) > p:
        gamma_hat = np.linalg.lstsq(X_arr[nz], np.log(y_arr[nz] + 0.1), rcond=None)[0]
    else:
        gamma_hat = np.zeros(p)
    # Logistic part: intercept only init
    delta_hat = np.zeros(p)
    delta_hat[0] = np.log(pi_hat / max(1 - pi_hat, 1e-10))

    prev_ll = -np.inf
    for iteration in range(max_iter):
        # E-step
        lam = np.exp(X_arr @ gamma_hat)
        lam = np.clip(lam, 1e-10, 1e6)
        logit_pi = X_arr @ delta_hat
        pi_i = 1.0 / (1.0 + np.exp(-logit_pi))
        pi_i = np.clip(pi_i, 1e-10, 1 - 1e-10)

        # Posterior probability of being from the zero component
        p_zero_poisson = np.exp(-lam)
        w = np.where(
            is_zero == 1,
            pi_i / (pi_i + (1 - pi_i) * p_zero_poisson),
            0.0,
        )
        w = np.clip(w, 1e-10, 1 - 1e-10)

        # M-step: logistic part (weighted logistic for zero-component membership)
        def neg_ll_logistic(delta):
            eta = X_arr @ delta
            pi_tmp = 1.0 / (1.0 + np.exp(-eta))
            pi_tmp = np.clip(pi_tmp, 1e-10, 1 - 1e-10)
            return -np.sum(w * np.log(pi_tmp) + (1 - w) * np.log(1 - pi_tmp))

        res_d = minimize(neg_ll_logistic, delta_hat, method="BFGS")
        delta_hat = res_d.x

        # M-step: Poisson part (weighted Poisson)
        def neg_ll_poisson(gamma):
            eta = X_arr @ gamma
            lam_tmp = np.exp(eta)
            lam_tmp = np.clip(lam_tmp, 1e-10, 1e6)
            return -np.sum((1 - w) * (y_arr * eta - lam_tmp))

        res_g = minimize(neg_ll_poisson, gamma_hat, method="BFGS")
        gamma_hat = res_g.x

        # Log-likelihood
        lam = np.exp(X_arr @ gamma_hat)
        lam = np.clip(lam, 1e-10, 1e6)
        logit_pi = X_arr @ delta_hat
        pi_i = 1.0 / (1.0 + np.exp(-logit_pi))
        pi_i = np.clip(pi_i, 1e-10, 1 - 1e-10)

        ll = 0.0
        for i in range(n):
            if y_arr[i] == 0:
                ll += np.log(pi_i[i] + (1 - pi_i[i]) * np.exp(-lam[i]))
            else:
                ll += np.log(1 - pi_i[i]) + y_arr[i] * np.log(lam[i]) - lam[i]
                ll -= sum(np.log(k) for k in range(1, int(y_arr[i]) + 1))

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # SE from Hessian
    def _extract_se(res_obj, n_params):
        if res_obj.hess_inv is not None:
            H = (
                np.asarray(res_obj.hess_inv)
                if not hasattr(res_obj.hess_inv, "todense")
                else np.asarray(res_obj.hess_inv.todense())
            )
            return np.sqrt(np.maximum(np.diag(H), 0.0))
        return np.full(n_params, np.nan)

    se_delta = _extract_se(res_d, p)
    se_gamma = _extract_se(res_g, p)

    names = ["intercept"] + list(x)
    z_d = delta_hat / np.where(se_delta > 0, se_delta, np.inf)
    z_g = gamma_hat / np.where(se_gamma > 0, se_gamma, np.inf)
    pv_d = 2.0 * (1.0 - norm.cdf(np.abs(z_d)))
    pv_g = 2.0 * (1.0 - norm.cdf(np.abs(z_g)))

    aic = 2.0 * (2 * p) - 2.0 * prev_ll

    return {
        "zero_coefficients": dict(zip(names, delta_hat.tolist())),
        "count_coefficients": dict(zip(names, gamma_hat.tolist())),
        "zero_se": dict(zip(names, se_delta.tolist())),
        "count_se": dict(zip(names, se_gamma.tolist())),
        "zero_p_values": dict(zip(names, pv_d.tolist())),
        "count_p_values": dict(zip(names, pv_g.tolist())),
        "aic": float(aic),
        "n": int(n),
    }


def cheatsheet() -> str:
    return "rey_zp({}) -> Zero-inflated Poisson regression via EM algorithm."
