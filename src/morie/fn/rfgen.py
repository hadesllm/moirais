# morie.fn — function file (hadesllm/morie)
"""Random forest for genomic prediction (Breiman 2001)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["random_forest_genomic"]


def _build_tree(X, y, rng, max_depth, min_samples, mtry):
    n, p = X.shape

    def split(idx, depth):
        if len(idx) < 2 * min_samples or depth >= max_depth:
            return {"leaf": True, "value": float(np.mean(y[idx]))}
        try_feats = rng.choice(p, size=min(mtry, p), replace=False)
        best = None
        cur_var = float(np.var(y[idx])) * len(idx)
        for f in try_feats:
            vals = np.unique(X[idx, f])
            if len(vals) < 2:
                continue
            mids = (vals[:-1] + vals[1:]) / 2.0
            for thr in mids:
                left = idx[X[idx, f] <= thr]
                right = idx[X[idx, f] > thr]
                if len(left) < min_samples or len(right) < min_samples:
                    continue
                ssE = (float(np.var(y[left])) * len(left)
                       + float(np.var(y[right])) * len(right))
                gain = cur_var - ssE
                if best is None or gain > best[0]:
                    best = (gain, f, thr, left, right)
        if best is None:
            return {"leaf": True, "value": float(np.mean(y[idx]))}
        _, f, thr, left, right = best
        return {"leaf": False, "feature": int(f), "threshold": float(thr),
                "left": split(left, depth + 1),
                "right": split(right, depth + 1)}

    return split(np.arange(n), 0)


def _predict_tree(node, X):
    if node["leaf"]:
        return np.full(X.shape[0], node["value"])
    mask = X[:, node["feature"]] <= node["threshold"]
    out = np.zeros(X.shape[0])
    if mask.any():
        out[mask] = _predict_tree(node["left"], X[mask])
    if (~mask).any():
        out[~mask] = _predict_tree(node["right"], X[~mask])
    return out


def random_forest_genomic(x, y, markers, n_trees: int = 100,
                          max_depth: int = 10, min_samples: int = 2,
                          mtry: int | None = None, seed: int = 0):
    """Random forest regression on the marker matrix.

    Uses scikit-learn's RandomForestRegressor if available; otherwise a
    NumPy CART implementation with bootstrap aggregation.

    Parameters
    ----------
    x : array-like (n,) or (n,q). Concatenated to markers.
    y : array-like (n,)
    markers : array-like (n,m)
    n_trees, max_depth, min_samples : standard RF hyperparameters.
    mtry : int, optional. Default sqrt(p).
    seed : int

    Returns
    -------
    RichResult with payload keys estimate, y_hat, oob_score,
    feature_importance, se, n, method.

    References
    ----------
    Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    Montesinos Lopez et al. (2022), Ch. 8.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1 and Xa.size > 0:
        Xa = Xa.reshape(-1, 1)
    feats = M if Xa.size == 0 else np.column_stack([Xa, M])
    p = feats.shape[1]
    if mtry is None:
        mtry = max(int(np.sqrt(p)), 1)
    method_used = "sklearn RandomForestRegressor"
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=n_trees, max_depth=max_depth,
            min_samples_split=min_samples, max_features=mtry,
            oob_score=True, random_state=seed, n_jobs=1,
        ).fit(feats, y)
        y_hat = rf.predict(feats)
        oob = float(rf.oob_score_) if hasattr(rf, "oob_score_") else float("nan")
        imp = rf.feature_importances_
    except Exception:
        method_used = "NumPy random-forest fallback"
        trees = []
        oob_preds = np.zeros(n)
        oob_count = np.zeros(n)
        imp = np.zeros(p)
        for _ in range(n_trees):
            boot = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool); oob_mask[boot] = False
            tree = _build_tree(feats[boot], y[boot], rng, max_depth,
                               min_samples, mtry)
            trees.append(tree)
            if oob_mask.any():
                preds = _predict_tree(tree, feats[oob_mask])
                oob_preds[oob_mask] += preds
                oob_count[oob_mask] += 1
        oob_count = np.where(oob_count == 0, 1, oob_count)
        oob_pred_avg = oob_preds / oob_count
        oob = float(1.0 - np.sum((y - oob_pred_avg) ** 2)
                    / max(np.sum((y - y.mean()) ** 2), 1e-12))
        y_hat = np.mean([_predict_tree(t, feats) for t in trees], axis=0)
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="Random-forest genomic predictor",
        summary_lines=[
            ("n", n),
            ("p (features)", p),
            ("n_trees", n_trees),
            ("max_depth", max_depth),
            ("mtry", mtry),
            ("OOB R^2", oob),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "oob_score": oob,
            "feature_importance": imp,
            "se": se,
            "n": n,
            "method": method_used,
        },
    )


def cheatsheet():
    return "rfgen: Random-forest genomic predictor"


# CANONICAL TEST
# np.random.seed(13); M = np.random.randn(40, 5)
# y = M[:,0] + 0.5*M[:,1]**2 + 0.2*np.random.randn(40)
# r = random_forest_genomic(np.zeros(40), y, M, seed=13); OOB R^2 > 0.
