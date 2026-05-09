# moirais.fn — function file (hadesllm/moirais)
"""Causal forest CATE estimator (Athey-Wager, honest trees).

Implements a simplified honest causal forest using random subsampling
and the R-learner residual-on-residual loss for asymptotically valid
confidence intervals on heterogeneous effects.

References
----------
Wager, S., & Athey, S. (2018). Estimation and inference of
heterogeneous treatment effects using random forests.
*Journal of the American Statistical Association*, 113(523), 1228-1242.

Athey, S., & Imbens, G. W. (2016). Recursive partitioning for
heterogeneous causal effects. *PNAS*, 113(27), 7353-7360.

Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
treatment effects. *Biometrika*, 108(2), 299-319.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm
from ._richresult import RichResult

__all__ = ["cfore"]


def cfore(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    n_trees: int = 100,
    min_leaf: int = 5,
    max_depth: int = 4,
    subsample_frac: float = 0.5,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate CATE via a simplified honest causal forest.

    Uses the R-learner decomposition:

    .. math::

        \hat{\tau}(x) \approx
        \frac{\mathbb{E}[(\tilde{Y} - \theta(X))\tilde{T} \mid X=x]}
             {\mathbb{E}[\tilde{T}^2 \mid X=x]}

    where :math:`\tilde{Y} = Y - m(X)`, :math:`\tilde{T} = T - e(X)`,
    and the forest aggregates predictions across subsampled trees.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    n_trees : int
        Number of trees in the forest.
    min_leaf : int
        Minimum observations per leaf (honesty constraint).
    max_depth : int
        Maximum tree depth.
    subsample_frac : float
        Fraction of data used per tree (honesty subsampling).
    seed : int
        Random seed for reproducibility.
    alpha : float
        Significance level for pointwise CIs.

    Returns
    -------
    dict
        ``cate``, ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.

    References
    ----------
    Wager & Athey (2018). JASA, 113(523), 1228-1242.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if not (len(T) == n == X.shape[0]):
        raise ValueError("Y, T, X must share first dimension.")
    if not (0.0 < subsample_frac < 1.0):
        raise ValueError("subsample_frac must be in (0, 1).")

    rng = np.random.default_rng(seed)
    Xa = np.column_stack([np.ones(n), X])

    # Nuisance: m(X) = E[Y|X] via OLS, e(X) = E[T|X] via logistic
    m_beta = np.linalg.lstsq(Xa, Y, rcond=None)[0]
    m_hat = Xa @ m_beta

    e_beta = _irls(Xa, T)
    e_hat = np.clip(expit(Xa @ e_beta), 0.01, 0.99)

    # R-learner residuals
    Y_res = Y - m_hat
    T_res = T - e_hat

    # Forest: aggregate CATE predictions across trees
    preds = np.zeros((n, n_trees))
    subsample_n = max(int(subsample_frac * n), 2 * min_leaf)

    for b in range(n_trees):
        idx = rng.choice(n, size=subsample_n, replace=False)
        X_b = X[idx]
        Yr_b = Y_res[idx]
        Tr_b = T_res[idx]

        # R-learner pseudo-outcome: w_i = Y_res_i / T_res_i (clipped)
        safe_T = np.where(np.abs(Tr_b) > 1e-6, Tr_b, np.sign(Tr_b) * 1e-6 + 1e-9)
        pseudo = Yr_b / safe_T

        # Fit regression tree on pseudo-outcomes
        tree = _fit_tree(X_b, pseudo, max_depth=max_depth, min_leaf=min_leaf)
        preds[:, b] = _predict_tree(tree, X)

    cate = np.mean(preds, axis=1)
    ate = float(np.mean(cate))
    # Variance via tree-level jackknife approximation
    tree_ates = np.mean(preds, axis=0)
    se = float(np.std(tree_ates, ddof=1) / np.sqrt(n_trees) * np.sqrt(n))
    se = max(se, float(np.std(cate, ddof=1) / np.sqrt(n)))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "cate": cate,
        "ate": ate,
        "se": se,
        "ci_lower": cate - z * np.std(preds, axis=1, ddof=1),
        "ci_upper": cate + z * np.std(preds, axis=1, ddof=1),
        "n": n,
        "method": "causal-forest",
    }


# ---- Minimal regression tree (axis-aligned splits) ----------------------

def _fit_tree(X, y, max_depth, min_leaf):
    """Recursively build a regression tree dict."""
    if max_depth == 0 or len(y) < 2 * min_leaf:
        return RichResult(payload={"leaf": True, "value": float(np.mean(y))})
    best = _best_split(X, y, min_leaf)
    if best is None:
        return RichResult(payload={"leaf": True, "value": float(np.mean(y))})
    feat, thr = best
    left = X[:, feat] <= thr
    right = ~left
    return {
        "leaf": False,
        "feat": feat,
        "thr": thr,
        "left": _fit_tree(X[left], y[left], max_depth - 1, min_leaf),
        "right": _fit_tree(X[right], y[right], max_depth - 1, min_leaf),
    }


def _best_split(X, y, min_leaf):
    n, p = X.shape
    best_loss = np.inf
    best = None
    for feat in range(p):
        xs = X[:, feat]
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = y[order]
        for i in range(min_leaf, n - min_leaf):
            if xs_s[i] == xs_s[i - 1]:
                continue
            yl, yr = ys_s[:i], ys_s[i:]
            loss = (np.var(yl) * len(yl) + np.var(yr) * len(yr)) / n
            if loss < best_loss:
                best_loss = loss
                best = (feat, (xs_s[i - 1] + xs_s[i]) / 2.0)
    return best


def _predict_tree(tree, X):
    if tree["leaf"]:
        return np.full(len(X), tree["value"])
    left = X[:, tree["feat"]] <= tree["thr"]
    out = np.empty(len(X))
    if left.any():
        out[left] = _predict_tree(tree["left"], X[left])
    if (~left).any():
        out[~left] = _predict_tree(tree["right"], X[~left])
    return out


def _irls(X, y, max_iter=25):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        A = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        b = (X.T * W) @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return beta


def cheatsheet() -> str:
    return "cfore(Y, T, X) -> Causal forest CATE (Wager & Athey 2018, JASA)."
