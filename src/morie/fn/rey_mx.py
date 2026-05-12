# morie.fn -- function file (hadesllm/morie)
"""Mixed-effects model (random intercept via EM algorithm)."""

import numpy as np
from scipy.stats import norm


def rey_mx(df, y: str = "y", x_fixed: list | str = "x", group_col: str = "group", max_iter: int = 200, tol: float = 1e-6, cdf=None) -> dict:
    r"""
    Linear mixed-effects model with random intercepts via EM.

    Fits the model:

    .. math::

        y_{ij} = X_{ij}\\beta + b_i + \\varepsilon_{ij}

    where :math:`b_i \\sim N(0, \\sigma^2_b)` and
    :math:`\\varepsilon_{ij} \\sim N(0, \\sigma^2_e)`.

    :param df: DataFrame with response, predictors, and group column.
    :param y: Response column name.
    :param x_fixed: Fixed-effect predictor column name(s).
    :param group_col: Column identifying the grouping factor.
    :param max_iter: Maximum EM iterations. Default 200.
    :param tol: Convergence tolerance on log-likelihood change. Default 1e-6.
    :return: dict with ``fixed_effects`` (dict), ``random_effects``
        (dict of group -> intercept), ``variance_components``
        (dict: sigma2_b, sigma2_e), ``aic``, ``n``, ``n_groups``.
    :raises ValueError: On missing columns.

    References
    ----------
    Laird, N. M. & Ware, J. H. (1982). Random-effects models for
    longitudinal data. *Biometrics*, 38(4), 963-974.

    Pinheiro, J. C. & Bates, D. M. (2000). Mixed-Effects Models in S
    and S-PLUS. Springer.
    """

    if isinstance(x_fixed, str):
        x_fixed = [x_fixed]
    for col in [y, group_col] + x_fixed:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    y_arr = np.asarray(df[y], dtype=float)
    X_arr = np.column_stack([np.ones(len(df))] + [np.asarray(df[c], dtype=float) for c in x_fixed])
    groups = np.asarray(df[group_col])
    unique_groups = np.unique(groups)
    n = len(y_arr)
    p = X_arr.shape[1]
    G = len(unique_groups)

    # Group indices
    group_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    # Initialize
    beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    resid = y_arr - X_arr @ beta
    sigma2_e = float(np.var(resid))
    sigma2_b = sigma2_e * 0.5
    b = {g: 0.0 for g in unique_groups}

    prev_ll = -np.inf
    for iteration in range(max_iter):
        # E-step: compute posterior mean and variance of b_i
        for g in unique_groups:
            idx = group_idx[g]
            n_i = len(idx)
            r_i = y_arr[idx] - X_arr[idx] @ beta
            # Posterior: b_i | y ~ N(mu_post, var_post)
            var_post = 1.0 / (n_i / sigma2_e + 1.0 / sigma2_b)
            mu_post = var_post * np.sum(r_i) / sigma2_e
            b[g] = mu_post

        # M-step: update beta
        y_adj = y_arr.copy()
        for g in unique_groups:
            idx = group_idx[g]
            y_adj[idx] -= b[g]
        beta = np.linalg.lstsq(X_arr, y_adj, rcond=None)[0]

        # Update variance components
        resid_all = np.zeros(n)
        for g in unique_groups:
            idx = group_idx[g]
            resid_all[idx] = y_arr[idx] - X_arr[idx] @ beta - b[g]
        sigma2_e = float(np.sum(resid_all**2) / n)
        sigma2_b = float(np.sum([b[g] ** 2 for g in unique_groups]) / G)
        sigma2_e = max(sigma2_e, 1e-10)
        sigma2_b = max(sigma2_b, 1e-10)

        # Log-likelihood (approximate)
        ll = -0.5 * n * np.log(2 * np.pi * sigma2_e)
        ll -= 0.5 * np.sum(resid_all**2) / sigma2_e
        ll -= 0.5 * G * np.log(2 * np.pi * sigma2_b)
        ll -= 0.5 * sum(b[g] ** 2 for g in unique_groups) / sigma2_b

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # SE for fixed effects (approximate: (X'X)^-1 * sigma2_e)
    XtX_inv = np.linalg.inv(X_arr.T @ X_arr + 1e-10 * np.eye(p))
    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2_e)
    z_vals = beta / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    names = ["intercept"] + list(x_fixed)
    n_params = p + 2  # beta + sigma2_e + sigma2_b
    aic = 2.0 * n_params - 2.0 * prev_ll

    return {
        "fixed_effects": dict(zip(names, beta.tolist())),
        "se": dict(zip(names, se_beta.tolist())),
        "p_values": dict(zip(names, p_vals.tolist())),
        "random_effects": {str(g): float(b[g]) for g in unique_groups},
        "variance_components": {
            "sigma2_b": float(sigma2_b),
            "sigma2_e": float(sigma2_e),
        },
        "aic": float(aic),
        "n": int(n),
        "n_groups": int(G),
    }


def cheatsheet() -> str:
    return "rey_mx({}) -> Mixed-effects model (random intercept via EM algorithm)."
