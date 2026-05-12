# morie.fn — function file (hadesllm/morie)
"""Ordinal logistic regression (proportional odds model)."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def rey_ol(df, y: str = "y", x: list | str = "x", alpha: float = 0.05, cdf=None) -> dict:
    r"""
    Ordinal logistic regression via the proportional odds model.

    For an ordered categorical response with *K* levels:

    .. math::

        \\log\\frac{P(Y \\le k)}{P(Y > k)}
        = \\alpha_k - X\\beta, \\quad k = 1, \\ldots, K-1

    :param df: DataFrame with response and predictor columns.
    :param y: Ordered categorical response column name.
    :param x: Predictor column name(s).
    :param alpha: Significance level. Default 0.05.
    :return: dict with ``thresholds`` (K-1 cutpoints), ``coefficients``,
        ``se``, ``p_values``, ``aic``, ``n``, ``categories``.
    :raises ValueError: On fewer than 2 categories or missing columns.

    References
    ----------
    McCullagh, P. (1980). Regression models for ordinal data.
    *Journal of the Royal Statistical Society: Series B*, 42(2), 109-142.

    Agresti, A. (2010). Analysis of Ordinal Categorical Data (2nd ed.).
    Wiley.
    """

    if isinstance(x, str):
        x = [x]
    for col in [y] + x:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame.")

    y_raw = np.asarray(df[y])
    categories = sorted(np.unique(y_raw))
    K = len(categories)
    if K < 2:
        raise ValueError(f"Need >= 2 ordered categories, got {K}.")

    # Map categories to 0..K-1
    cat_map = {c: i for i, c in enumerate(categories)}
    y_int = np.array([cat_map[v] for v in y_raw])

    X_arr = np.column_stack([np.asarray(df[c], dtype=float) for c in x])
    n, p = X_arr.shape
    n_thresholds = K - 1

    def neg_loglik(params):
        thresholds = params[:n_thresholds]
        beta = params[n_thresholds:]
        Xb = X_arr @ beta
        ll = 0.0
        for i in range(n):
            k = y_int[i]
            if k == 0:
                # P(Y = 0) = logistic(alpha_0 - Xb)
                eta = thresholds[0] - Xb[i]
                p_i = 1.0 / (1.0 + np.exp(-eta))
            elif k == K - 1:
                # P(Y = K-1) = 1 - logistic(alpha_{K-2} - Xb)
                eta = thresholds[K - 2] - Xb[i]
                p_i = 1.0 - 1.0 / (1.0 + np.exp(-eta))
            else:
                eta_hi = thresholds[k] - Xb[i]
                eta_lo = thresholds[k - 1] - Xb[i]
                p_i = 1.0 / (1.0 + np.exp(-eta_hi)) - 1.0 / (1.0 + np.exp(-eta_lo))
            p_i = max(p_i, 1e-15)
            ll += np.log(p_i)
        return -ll

    # Initialize thresholds as evenly spaced quantiles
    thresh0 = np.array([norm.ppf((k + 1) / K) for k in range(n_thresholds)])
    beta0 = np.zeros(p)
    x0 = np.concatenate([thresh0, beta0])

    result = minimize(neg_loglik, x0, method="BFGS")
    params_hat = result.x
    thresh_hat = params_hat[:n_thresholds]
    beta_hat = params_hat[n_thresholds:]

    # SE from Hessian
    if result.hess_inv is not None:
        H_inv = (
            np.asarray(result.hess_inv)
            if not hasattr(result.hess_inv, "todense")
            else np.asarray(result.hess_inv.todense())
        )
        se_all = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
    else:
        se_all = np.full(len(params_hat), np.nan)

    se_thresh = se_all[:n_thresholds]
    se_beta = se_all[n_thresholds:]
    z_vals = beta_hat / np.where(se_beta > 0, se_beta, np.inf)
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))

    loglik = -result.fun
    n_params_total = n_thresholds + p
    aic = 2.0 * n_params_total - 2.0 * loglik

    names = list(x)

    return {
        "thresholds": dict(
            zip(
                [f"alpha_{k}" for k in range(n_thresholds)],
                thresh_hat.tolist(),
            )
        ),
        "threshold_se": dict(
            zip(
                [f"alpha_{k}" for k in range(n_thresholds)],
                se_thresh.tolist(),
            )
        ),
        "coefficients": dict(zip(names, beta_hat.tolist())),
        "se": dict(zip(names, se_beta.tolist())),
        "p_values": dict(zip(names, p_vals.tolist())),
        "aic": float(aic),
        "n": int(n),
        "categories": [str(c) for c in categories],
    }


def cheatsheet() -> str:
    return "rey_ol({}) -> Ordinal logistic regression (proportional odds model)."
