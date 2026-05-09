# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""An unexamined life is not worth living. — Socrates"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def _best_stump(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    """Find best decision stump (feature, threshold, polarity)."""
    n, p = X.shape
    best_err = np.inf
    best = (0, 0.0, 1)
    for j in range(p):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            for polarity in (1, -1):
                pred = np.where(polarity * X[:, j] < polarity * t, -1, 1)
                err = float(np.sum(w * (pred != y)))
                if err < best_err:
                    best_err = err
                    best = (j, float(t), polarity)
    return best


def adaboost(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
) -> DescriptiveResult:
    """AdaBoost with decision stumps.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
        Labels in {-1, +1}.
    n_estimators : int, default 50

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if set(np.unique(y)) - {-1.0, 1.0}:
        labels = np.unique(y)
        if len(labels) == 2:
            y = np.where(y == labels[0], -1.0, 1.0)
    n = len(y)
    w = np.full(n, 1 / n)
    stumps = []
    alphas = []

    for _ in range(n_estimators):
        feat, thresh, pol = _best_stump(X, y, w)
        pred = np.where(pol * X[:, feat] < pol * thresh, -1.0, 1.0)
        err = float(np.sum(w * (pred != y)))
        err = np.clip(err, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(-alpha * y * pred)
        w /= w.sum()
        stumps.append((feat, thresh, pol))
        alphas.append(alpha)

    alphas_arr = np.array(alphas)
    final_pred = np.zeros(n)
    for alpha, (feat, thresh, pol) in zip(alphas_arr, stumps):
        final_pred += alpha * np.where(pol * X[:, feat] < pol * thresh, -1.0, 1.0)
    predictions = np.sign(final_pred)
    accuracy = float(np.mean(predictions == y))

    return DescriptiveResult(
        name="AdaBoost",
        value=accuracy,
        extra={
            "predictions": predictions,
            "weights": alphas_arr,
            "errors": [
                float(np.sum(w * (np.where(s[2] * X[:, s[0]] < s[2] * s[1], -1, 1) != y)))
                for s, w_dummy in zip(stumps, [np.full(n, 1 / n)])
            ],
            "n_estimators": n_estimators,
            "n": n,
        },
    )


ada = adaboost


def cheatsheet() -> str:
    return "_best_stump({}) -> AdaBoost with decision stumps (pure numpy). 'Power! Unlimite"
