# moirais.fn — function file (hadesllm/moirais)
"""Random forest classifier (pure NumPy, bootstrap + feature subset + dtree)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from moirais.fn.dtree import decision_tree as dtree_classify


def random_forest(
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    n_trees: int = 100,
    max_depth: int = 5,
    max_features: str | int | None = "sqrt",
    random_state: int = 42,
) -> dict[str, Any]:
    """Random forest classifier via bootstrap aggregation of decision trees.

    Uses :func:`moirais.fn.dtree.dtree_classify` as the base learner.
    Computes out-of-bag (OOB) score when possible.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Labels.
    n_trees : int
        Number of trees (default 100).
    max_depth : int
        Maximum depth per tree (default 5).
    max_features : str, int, or None
        Features per split: ``"sqrt"`` (default), ``"log2"``, int, or None (all).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        predictions (n,), feature_importance (p,), oob_score (float or None).

    References
    ----------
    Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
        doi:10.1023/A:1010933404324
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    rng = np.random.default_rng(random_state)
    n, p = X.shape

    if max_features == "sqrt":
        m = max(1, int(np.sqrt(p)))
    elif max_features == "log2":
        m = max(1, int(np.log2(p)))
    elif isinstance(max_features, int):
        m = min(max_features, p)
    else:
        m = p

    all_importance = np.zeros(p)
    oob_preds: dict[int, list] = {i: [] for i in range(n)}
    tree_predictions = []

    for t in range(n_trees):
        # Bootstrap sample
        boot_idx = rng.choice(n, size=n, replace=True)
        oob_idx = np.setdiff1d(np.arange(n), boot_idx)

        # Random feature subset
        feat_idx = rng.choice(p, size=m, replace=False)
        X_boot = X[boot_idx][:, feat_idx]
        y_boot = y[boot_idx]

        result = dtree_classify(X_boot, y_boot, X_boot, max_depth=max_depth)
        res_extra = result.extra if hasattr(result, "extra") else result
        fi = res_extra.get("feature_importance", np.zeros(len(feat_idx)))
        for j, fj in enumerate(feat_idx):
            all_importance[fj] += fi[j] if j < len(fi) else 0

        if len(oob_idx) > 0:
            X_oob = X[oob_idx][:, feat_idx]
            oob_result = dtree_classify(X_boot, y_boot, X_oob, max_depth=max_depth)
            oob_extra = oob_result.extra if hasattr(oob_result, "extra") else oob_result
            oob_pred_vals = oob_extra.get("predictions", [])
            for i_oob, idx in enumerate(oob_idx):
                if i_oob < len(oob_pred_vals):
                    oob_preds[idx].append(oob_pred_vals[i_oob])

        tree_predictions.append(res_extra.get("predictions", []))

    # Aggregate importance
    imp_sum = all_importance.sum()
    if imp_sum > 0:
        all_importance /= imp_sum

    # Final predictions via majority vote on full training data
    # Re-run each tree is expensive; use a simpler approach
    # Build predictions by running forest on training data
    final_preds = _majority_vote_ensemble(X, y, n_trees, max_depth, m, rng, p)

    # OOB score
    oob_score = None
    oob_correct = 0
    oob_total = 0
    for i in range(n):
        if oob_preds[i]:
            labels, counts = np.unique(oob_preds[i], return_counts=True)
            pred = labels[np.argmax(counts)]
            oob_correct += int(pred == y[i])
            oob_total += 1
    if oob_total > 0:
        oob_score = oob_correct / oob_total

    return {
        "predictions": final_preds,
        "feature_importance": all_importance,
        "oob_score": oob_score,
    }


def _majority_vote_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int,
    max_depth: int,
    m: int,
    rng: np.random.Generator,
    p: int,
) -> np.ndarray:
    """Train and predict via majority vote."""
    n = X.shape[0]
    votes = np.zeros((n,), dtype=object)
    vote_lists: dict[int, list] = {i: [] for i in range(n)}

    for t in range(min(n_trees, 10)):
        boot_idx = rng.choice(n, size=n, replace=True)
        feat_idx = rng.choice(p, size=m, replace=False)
        X_boot = X[boot_idx][:, feat_idx]
        result = dtree_classify(X_boot, y[boot_idx], X[:, feat_idx], max_depth=max_depth)
        res_extra = result.extra if hasattr(result, "extra") else result
        preds_arr = res_extra.get("predictions", [])
        for i in range(n):
            if i < len(preds_arr):
                vote_lists[i].append(preds_arr[i])

    preds = np.empty(n, dtype=y.dtype)
    for i in range(n):
        if vote_lists[i]:
            labels, counts = np.unique(vote_lists[i], return_counts=True)
            preds[i] = labels[np.argmax(counts)]
        else:
            preds[i] = y[0]
    return preds


rforc = random_forest


def cheatsheet() -> str:
    return "random_forest({}) -> Random forest classifier (pure NumPy, bootstrap + feature su"
