# morie.fn -- function file (hadesllm/morie)
"""Sequential forward feature selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The road up and the road down are the same thing. -- Heraclitus"


def forward_select(X, y, max_features=None, cv=3, **kwargs) -> DescriptiveResult:
    """Sequential forward feature selection using CV accuracy.

    Greedily adds features one at a time, selecting the feature that
    maximises cross-validated nearest-centroid accuracy.

    Parameters
    ----------
    X : array-like of shape (n, p)
    y : array-like of shape (n,)
    max_features : int or None
        Maximum features to select. Default: p.
    cv : int
        Cross-validation folds (default 3).

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    n, p = X.shape
    if max_features is None:
        max_features = p
    max_features = min(max_features, p)

    rng = np.random.default_rng(kwargs.get("seed", 42))
    indices = rng.permutation(n)
    fold_size = n // cv

    def _cv_accuracy(feat_set):
        accs = []
        for fold in range(cv):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            tr_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])
            X_tr = X[tr_idx][:, feat_set]
            y_tr = y[tr_idx]
            X_val = X[val_idx][:, feat_set]
            y_val = y[val_idx]
            classes = np.unique(y_tr)
            centroids = np.array([X_tr[y_tr == c].mean(axis=0) for c in classes])
            dists = np.array([np.sum((X_val - c) ** 2, axis=1) for c in centroids])
            preds = classes[np.argmin(dists, axis=0)]
            accs.append(float(np.mean(preds == y_val)))
        return np.mean(accs)

    selected = []
    remaining = list(range(p))
    scores_history = []

    for _ in range(max_features):
        best_score = -1.0
        best_feat = remaining[0]
        for f in remaining:
            candidate = selected + [f]
            score = _cv_accuracy(candidate)
            if score > best_score:
                best_score = score
                best_feat = f
        selected.append(best_feat)
        remaining.remove(best_feat)
        scores_history.append(best_score)
        if best_score >= 1.0:
            break

    return DescriptiveResult(
        name="forward_select",
        value=float(scores_history[-1]) if scores_history else 0.0,
        extra={
            "selected_features": np.array(selected),
            "scores_history": np.array(scores_history),
            "n_selected": len(selected),
            "best_accuracy": float(scores_history[-1]) if scores_history else 0.0,
        },
    )


fwsel = forward_select


def cheatsheet() -> str:
    return "forward_select({}) -> Sequential forward feature selection."
