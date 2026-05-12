# morie.fn — function file (hadesllm/morie)
"""Ordered logit (proportional odds) model AND
Jonckheere-Terpstra ordered-alternatives test (Gibbons Ch 10.6).

Two unrelated callables share this module because of a name
collision in the Gibbons spec; both are exported via ``__all__``.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from ._containers import RegressionResult
from ._richresult import RichResult

__all__ = ["ordered_logit", "ordered_alternatives_test"]


def ordered_logit(
    y: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 200,
) -> RegressionResult:
    r"""Ordered logit (proportional odds) via MLE.

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


def ordered_alternatives_test(groups):
    """Jonckheere-Terpstra test for ordered alternatives (Gibbons Ch 10.6).

    Tests H0: F_1 = F_2 = ... = F_k against the *ordered* alternative
    H1: F_1 <= F_2 <= ... <= F_k (with at least one strict inequality).
    Groups must be supplied in the order specified by H1.

    Statistic J = sum_{i<j} U_{ij} where U_{ij} = #{(x in G_i, y in G_j) : x < y}
    + 0.5 * #{ties}.

    Parameters
    ----------
    groups : sequence of array-like
        Samples in monotone-hypothesised order.

    Returns
    -------
    RichResult with payload:
        statistic, p_value, z, n, k, method
    """
    arrs = [np.asarray(g, dtype=float).ravel() for g in groups]
    k = len(arrs)
    if k < 2 or any(a.size < 1 for a in arrs):
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "z": np.nan,
            "n": 0, "k": k,
            "method": "Jonckheere-Terpstra ordered-alternatives test",
        })
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            xi = arrs[i][:, None]
            yj = arrs[j][None, :]
            J += float(np.sum(xi < yj) + 0.5 * np.sum(xi == yj))
    ns = np.array([a.size for a in arrs], dtype=float)
    N = float(ns.sum())
    E_J = (N ** 2 - np.sum(ns ** 2)) / 4.0
    Var_J = (N ** 2 * (2 * N + 3) - np.sum(ns ** 2 * (2 * ns + 3))) / 72.0
    z = (J - E_J) / np.sqrt(Var_J) if Var_J > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return RichResult(payload={
        "statistic": float(J),
        "p_value": float(p),
        "z": float(z),
        "E_J": float(E_J),
        "Var_J": float(Var_J),
        "n": int(N),
        "k": k,
        "method": "Jonckheere-Terpstra ordered-alternatives test",
    })


# CANONICAL TEST (Jonckheere)
# >>> ordered_alternatives_test([[1,2,3,4], [3,4,5,6], [5,6,7,8]])
# Monotone increase across 3 groups -> J large, z positive, p small


def cheatsheet() -> str:
    return ("ordered_logit({}) -> Ordered logit (proportional odds) model. "
            "ordered_alternatives_test([groups]) -> Jonckheere-Terpstra test.")
