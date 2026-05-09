# moirais.fn — function file (hadesllm/moirais)
"""Minimum Redundancy Maximum Relevance (mRMR) scoring."""

import numpy as np

from ._containers import DescriptiveResult
def mrmr_score(X, y, n_features: int = 10, **kwargs) -> DescriptiveResult:
    """
    Greedy mRMR feature selection using mutual-information proxies.

    Uses F-statistic as relevance proxy and absolute Pearson
    correlation as redundancy proxy (Ding & Peng, 2005).

    :param X: (n, d) data matrix.
    :param y: (n,) target (continuous or discrete).
    :param n_features: Number of features to select.
    :return: DescriptiveResult with selected indices and mRMR scores.

    References
    ----------
    Peng H, Long F, Ding C (2005). Feature selection based on
    mutual information: criteria of max-dependency, max-relevance,
    and min-redundancy. IEEE TPAMI, 27(8), 1226-1238.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    n_features = min(n_features, d)
    relevance = np.array([abs(float(np.corrcoef(X[:, j], y)[0, 1])) if np.std(X[:, j]) > 0 else 0.0 for j in range(d)])
    selected = []
    remaining = list(range(d))
    scores = []
    for _ in range(n_features):
        best_score = -np.inf
        best_idx = remaining[0]
        for j in remaining:
            rel = relevance[j]
            if len(selected) == 0:
                red = 0.0
            else:
                red = np.mean(
                    [
                        abs(float(np.corrcoef(X[:, j], X[:, s])[0, 1]))
                        if np.std(X[:, j]) > 0 and np.std(X[:, s]) > 0
                        else 0.0
                        for s in selected
                    ]
                )
            score = rel - red
            if score > best_score:
                best_score = score
                best_idx = j
        selected.append(best_idx)
        scores.append(float(best_score))
        remaining.remove(best_idx)
    return DescriptiveResult(
        name="mrmr_score",
        value=float(len(selected)),
        extra={
            "selected_indices": selected,
            "mrmr_scores": scores,
            "n_features": n_features,
            "n_total": d,
        },
    )


mirmf = mrmr_score


def cheatsheet() -> str:
    return "mrmr_score({}) -> Minimum Redundancy Maximum Relevance (mRMR) scoring."
