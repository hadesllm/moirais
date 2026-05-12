# morie.fn -- function file (hadesllm/morie)
"""Multinomial logit regression."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._containers import RegressionResult


def multinomial_logit(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    max_iter: int = 200,
) -> RegressionResult:
    """Multinomial logistic regression via MLE (BFGS).

    Reference category is the smallest label (coded as 0 internally).
    Estimates J-1 sets of coefficients where J is the number of categories.

    Parameters
    ----------
    y : (n,) integer category labels (0, 1, ..., J-1)
    X : (n, p) predictors
    add_intercept : bool
    max_iter : int

    Returns
    -------
    RegressionResult

    References
    ----------
    McFadden, D. (1973). Conditional logit analysis of qualitative choice
    behavior. In *Frontiers in Econometrics*, ed. P. Zarembka.
    """
    y = np.asarray(y, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    cats = np.sort(np.unique(y))
    J = len(cats)
    if J < 3:
        raise ValueError("Use logistic regression for binary outcomes.")
    cat_map = {c: j for j, c in enumerate(cats)}
    y_int = np.array([cat_map[yi] for yi in y])

    n_params = (J - 1) * k

    def _softmax(X, beta_mat):
        logits = X @ beta_mat
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        denom = 1.0 + exp_l.sum(axis=1, keepdims=True)
        probs = np.column_stack([1.0 / denom.ravel(), exp_l / denom])
        return probs

    def neg_loglik(params):
        beta_mat = params.reshape(k, J - 1)
        probs = _softmax(X, beta_mat)
        probs = np.clip(probs, 1e-300, None)
        ll = np.sum(np.log(probs[np.arange(n), y_int]))
        return -ll

    def grad(params):
        beta_mat = params.reshape(k, J - 1)
        probs = _softmax(X, beta_mat)
        g = np.zeros((k, J - 1))
        for j in range(1, J):
            indicator = (y_int == j).astype(float)
            g[:, j - 1] = X.T @ (indicator - probs[:, j])
        return -g.ravel()

    x0 = np.zeros(n_params)
    res = optimize.minimize(neg_loglik, x0, jac=grad, method="BFGS",
                            options={"maxiter": max_iter})
    beta_mat = res.x.reshape(k, J - 1)

    probs = _softmax(X, beta_mat)
    y_pred = cats[np.argmax(probs, axis=1)]
    accuracy = float(np.mean(y_pred == y))

    ll = float(-res.fun)
    aic = -2 * ll + 2 * n_params

    names_base = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    coef_dict = {}
    for j in range(J - 1):
        cat_label = str(cats[j + 1])
        for i, nm in enumerate(names_base):
            coef_dict[f"cat{cat_label}_{nm}"] = float(beta_mat[i, j])

    return RegressionResult(
        method="Multinomial Logit",
        coefficients=coef_dict,
        se={nm: float("nan") for nm in coef_dict},
        p_values={nm: float("nan") for nm in coef_dict},
        fitted=probs,
        residuals=None,
        n=n,
        k=n_params,
        extra={
            "categories": cats.tolist(),
            "ref_category": int(cats[0]),
            "accuracy": accuracy,
            "log_likelihood": ll,
            "aic": aic,
        },
    )


multi = multinomial_logit


def cheatsheet() -> str:
    return "multinomial_logit({}) -> Multinomial logit regression."
