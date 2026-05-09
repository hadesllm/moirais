"""Simplified XGBoost (L2 regularized gradient boosting)."""

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult


def _xgb_build(X, g, h, depth, max_depth, lam, min_child):
    n = len(g)
    if depth >= max_depth or n <= min_child:
        return RichResult(payload={"leaf": True, "value": float(-g.sum() / (h.sum() + lam))})
    best_gain = 0.0
    best_feat, best_thr = 0, 0.0
    G, H = g.sum(), h.sum()
    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        GL, HL = 0.0, 0.0
        sorted_idx = np.argsort(X[:, j])
        for idx in sorted_idx:
            GL += g[idx]
            HL += h[idx]
            GR, HR = G - GL, H - HL
            if min_child > HL or min_child > HR:
                continue
            gain = (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)) / 2
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thr = float(X[idx, j])
    if best_gain <= 0:
        return RichResult(payload={"leaf": True, "value": float(-G / (H + lam))})
    left = X[:, best_feat] <= best_thr
    right = ~left
    return {
        "leaf": False,
        "feature": best_feat,
        "threshold": best_thr,
        "left": _xgb_build(X[left], g[left], h[left], depth + 1, max_depth, lam, min_child),
        "right": _xgb_build(X[right], g[right], h[right], depth + 1, max_depth, lam, min_child),
    }


def _xgb_pred(tree, x):
    if tree["leaf"]:
        return tree["value"]
    return _xgb_pred(tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"], x)


def xgboost_simple(
    X: np.ndarray, y: np.ndarray, n_trees: int = 100, learning_rate: float = 0.1, max_depth: int = 3, lam: float = 1.0
) -> DescriptiveResult:
    """
    Simplified XGBoost with L2 regularisation (pure numpy).

    :param X: (n, p) feature matrix.
    :param y: (n,) target.
    :param n_trees: Number of boosting rounds.
    :param learning_rate: Shrinkage.
    :param max_depth: Max tree depth.
    :param lam: L2 regularisation on leaf weights.
    :return: DescriptiveResult with predictions and R-squared.

    References
    ----------
    Chen T, Guestrin C (2016). XGBoost: a scalable tree boosting system.
    KDD, 785-794.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    f = np.full(n, y.mean())
    for _ in range(n_trees):
        g = f - y
        h = np.ones(n)
        tree = _xgb_build(X, g, h, 0, max_depth, lam, 1)
        update = np.array([_xgb_pred(tree, x) for x in X])
        f += learning_rate * update
    ss_res = np.sum((y - f) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="xgboost_simple",
        value=r2,
        extra={"predictions": f, "r_squared": r2, "n_trees": n_trees, "learning_rate": learning_rate, "n": n},
    )


xgb_ = xgboost_simple


def cheatsheet() -> str:
    return "_xgb_build({}) -> Simplified XGBoost (L2 regularized gradient boosting)."
