# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap aggregating (bagging) classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Knowing yourself is the beginning of all wisdom. — Aristotle"


def bagging_classify(X_train, y_train, X_test, n_estimators=20, max_depth=5, **kwargs) -> DescriptiveResult:
    """Bagging classifier using decision tree base learners.

    Each base learner is trained on a bootstrap sample of the training data.
    Final prediction is majority vote.

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    n_estimators : int
        Number of bootstrap models (default 20).
    max_depth : int
        Max depth per tree (default 5).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123--140.
    """
    from .dtree import decision_tree

    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    n = X_tr.shape[0]
    rng = np.random.default_rng(kwargs.get("seed", 42))

    all_preds = []
    oob_correct = 0
    oob_total = 0

    for _ in range(n_estimators):
        idx = rng.integers(0, n, size=n)
        oob_idx = np.setdiff1d(np.arange(n), idx)
        res = decision_tree(X_tr[idx], y[idx], X_te, max_depth=max_depth)
        all_preds.append(res.extra["predictions"])
        if len(oob_idx) > 0:
            oob_res = decision_tree(X_tr[idx], y[idx], X_tr[oob_idx], max_depth=max_depth)
            oob_correct += np.sum(oob_res.extra["predictions"] == y[oob_idx])
            oob_total += len(oob_idx)

    all_preds = np.array(all_preds)
    predictions = np.array([np.bincount(all_preds[:, j].astype(int)).argmax() for j in range(len(X_te))])
    oob_acc = float(oob_correct / oob_total) if oob_total > 0 else float("nan")

    return DescriptiveResult(
        name="bagging_classify",
        value=oob_acc,
        extra={
            "predictions": predictions,
            "oob_accuracy": oob_acc,
            "n_estimators": n_estimators,
        },
    )


bgcls = bagging_classify


def cheatsheet() -> str:
    return "bagging_classify({}) -> Bootstrap aggregating (bagging) classifier."
