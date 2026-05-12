# morie.fn — function file (hadesllm/morie)
"""Gradient boosting for genomic prediction (Friedman 2001)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gradient_boosting_genomic"]


def _stump_split(X, r):
    """Best regression stump (depth-1 tree) for residual `r`."""
    n, p = X.shape
    cur_var = float(np.var(r)) * n
    best = None
    for f in range(p):
        order = np.argsort(X[:, f])
        sx = X[order, f]
        sr = r[order]
        cumsum = np.cumsum(sr)
        total = cumsum[-1]
        for k in range(1, n):
            if sx[k - 1] == sx[k]:
                continue
            left_sum = cumsum[k - 1]
            right_sum = total - left_sum
            n_l = k; n_r = n - k
            sse_l = float(np.var(sr[:k])) * n_l
            sse_r = float(np.var(sr[k:])) * n_r
            sse = sse_l + sse_r
            gain = cur_var - sse
            if best is None or gain > best[0]:
                thr = 0.5 * (sx[k - 1] + sx[k])
                left_val = left_sum / n_l
                right_val = right_sum / n_r
                best = (gain, f, thr, left_val, right_val)
    if best is None:
        return None
    return {"feature": int(best[1]), "threshold": float(best[2]),
            "left_val": float(best[3]), "right_val": float(best[4])}


def _stump_predict(stump, X):
    if stump is None:
        return np.zeros(X.shape[0])
    out = np.where(X[:, stump["feature"]] <= stump["threshold"],
                   stump["left_val"], stump["right_val"])
    return out


def gradient_boosting_genomic(x, y, markers, n_estimators: int = 100,
                              learning_rate: float = 0.1,
                              max_depth: int = 3, seed: int = 0):
    """Gradient-boosted regression on markers (squared-error loss).

    Tries scikit-learn's GradientBoostingRegressor; falls back to a NumPy
    boosted-stumps implementation (depth-1 trees by default).

    Parameters
    ----------
    x : array-like (n,) or (n,q). Concatenated to markers.
    y : array-like (n,)
    markers : array-like (n, m)
    n_estimators : int, default 100.
    learning_rate : float, default 0.1.
    max_depth : int, default 3 (sklearn only; fallback uses stumps).
    seed : int

    Returns
    -------
    RichResult with payload keys estimate, y_hat, train_loss, se, n, method.

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: a gradient
        boosting machine. Annals of Statistics, 29(5), 1189-1232.
    Montesinos Lopez et al. (2022), Ch. 9.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1 and Xa.size > 0:
        Xa = Xa.reshape(-1, 1)
    feats = M if Xa.size == 0 else np.column_stack([Xa, M])
    method_used = "sklearn GradientBoostingRegressor"
    train_loss = []
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=seed,
        ).fit(feats, y)
        y_hat = gb.predict(feats)
        train_loss = list(gb.train_score_) if hasattr(gb, "train_score_") else []
    except Exception:
        method_used = "NumPy gradient-boosting stumps fallback"
        F = np.full(n, float(np.mean(y)))
        stumps = []
        for _ in range(n_estimators):
            r = y - F
            st = _stump_split(feats, r)
            stumps.append(st)
            F = F + learning_rate * _stump_predict(st, feats)
            train_loss.append(float(np.mean((y - F) ** 2)))
        y_hat = F
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="Gradient-boosting genomic predictor",
        summary_lines=[
            ("n", n),
            ("p (features)", feats.shape[1]),
            ("n_estimators", n_estimators),
            ("learning_rate", learning_rate),
            ("max_depth", max_depth),
            ("final train MSE",
             train_loss[-1] if train_loss else float(np.mean(resid ** 2))),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "train_loss": np.asarray(train_loss),
            "se": se,
            "n": n,
            "method": method_used,
        },
    )


def cheatsheet():
    return "gbgen: Gradient-boosting genomic predictor"


# CANONICAL TEST
# np.random.seed(14); M = np.random.randn(40, 4)
# y = np.sign(M[:,0]) + 0.3*np.random.randn(40)
# r = gradient_boosting_genomic(np.zeros(40), y, M, seed=14); train MSE decreasing.
