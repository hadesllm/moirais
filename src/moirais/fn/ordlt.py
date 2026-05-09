# moirais.fn — function file (hadesllm/moirais)
"""Ordered logit (proportional odds) model."""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from ._containers import RegressionResult


def ordered_logit(
    y: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 200,
) -> RegressionResult:
    """Ordered logit (proportional odds) via MLE.

    Estimates cutpoints :math:`\\alpha_1 < \\alpha_2 < \\dots < \\alpha_{J-1}`
    and a common coefficient vector :math:`\\beta` such that
    :math:`P(Y \\le j) = \\text{logit}^{-1}(\\alpha_j - X\\beta)`.

    Parameters
    ----------
    y : (n,) ordinal categories (0, 1, ..., J-1)
    X : (n, p) predictors (no intercept -- absorbed by cutpoints)
    max_iter : int

    Returns
    -------
    RegressionResult

    References
    ----------
    McCullagh, P. (1980). Regression models for ordinal data. *JRSS-B*,
    42(2), 109--142.
    """
    y = np.asarray(y, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    cats = np.sort(np.unique(y))
    J = len(cats)
    if J < 3:
        raise ValueError("Need >= 3 ordered categories.")
    cat_map = {c: j for j, c in enumerate(cats)}
    y_int = np.array([cat_map[yi] for yi in y])
    n_cut = J - 1

    def _cum_prob(alpha, beta):
        eta = X @ beta
        cp = np.zeros((n, J + 1))
        cp[:, 0] = 0.0
        cp[:, J] = 1.0
        for j in range(n_cut):
            cp[:, j + 1] = special.expit(alpha[j] - eta)
        return cp

    def neg_loglik(params):
        raw_alpha = params[:n_cut]
        alpha = np.cumsum(np.concatenate([[raw_alpha[0]], np.exp(raw_alpha[1:])]))
        beta = params[n_cut:]
        cp = _cum_prob(alpha, beta)
        probs = np.diff(cp, axis=1)
        probs = np.clip(probs, 1e-300, None)
        ll = np.sum(np.log(probs[np.arange(n), y_int]))
        return -ll

    alpha0 = np.array([float(j) for j in range(n_cut)])
    raw_alpha0 = np.concatenate([[alpha0[0]], np.log(np.diff(alpha0) + 0.1)])
    x0 = np.concatenate([raw_alpha0, np.zeros(p)])

    res = optimize.minimize(neg_loglik, x0, method="BFGS",
                            options={"maxiter": max_iter})
    raw_alpha = res.x[:n_cut]
    alpha = np.cumsum(np.concatenate([[raw_alpha[0]], np.exp(raw_alpha[1:])]))
    beta = res.x[n_cut:]

    ll = float(-res.fun)
    aic = -2 * ll + 2 * (n_cut + p)

    cp = _cum_prob(alpha, beta)
    probs = np.diff(cp, axis=1)
    y_pred = cats[np.argmax(probs, axis=1)]
    accuracy = float(np.mean(y_pred == y))

    coef_dict = {}
    for j in range(n_cut):
        coef_dict[f"alpha_{j + 1}|{j + 2}"] = float(alpha[j])
    for j in range(p):
        coef_dict[f"x{j}"] = float(beta[j])

    return RegressionResult(
        method="Ordered Logit",
        coefficients=coef_dict,
        se={nm: float("nan") for nm in coef_dict},
        p_values={nm: float("nan") for nm in coef_dict},
        fitted=probs,
        residuals=None,
        n=n,
        k=n_cut + p,
        extra={
            "categories": cats.tolist(),
            "cutpoints": alpha.tolist(),
            "accuracy": accuracy,
            "log_likelihood": ll,
            "aic": aic,
        },
    )


ordlt = ordered_logit


def cheatsheet() -> str:
    return "ordered_logit({}) -> Ordered logit (proportional odds) model."
