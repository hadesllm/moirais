# morie.fn -- function file (hadesllm/morie)
"""Isolation-forest-style anomaly scoring."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def anomaly_isolation(
    data: np.ndarray,
    *,
    n_trees: int = 100,
    subsample: int = 256,
    contamination: float = 0.1,
    seed: int = 42,
) -> DescriptiveResult:
    """Isolation-forest-style anomaly scoring.

    Implements a simplified isolation forest (Liu et al., 2008). Each tree
    recursively splits on random features/thresholds until each point is
    isolated. Anomalies have shorter average path lengths.

    Parameters
    ----------
    data : ndarray of shape (n, p)
        Input data.
    n_trees : int
        Number of isolation trees.
    subsample : int
        Subsample size per tree.
    contamination : float
        Expected fraction of anomalies (for threshold).
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = anomaly scores (ndarray, higher = more anomalous)
        and ``extra`` containing labels and threshold.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    max_depth = int(np.ceil(np.log2(max(subsample, 2))))

    def _c(size):
        if size <= 1:
            return 0.0
        return 2.0 * (np.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size

    def _build_tree(subset, depth=0):
        n_sub = len(subset)
        if depth >= max_depth or n_sub <= 1:
            return n_sub
        feat = rng.integers(0, p)
        lo, hi = subset[:, feat].min(), subset[:, feat].max()
        if lo == hi:
            return n_sub
        split = rng.uniform(lo, hi)
        left_mask = subset[:, feat] < split
        return (feat, split, _build_tree(subset[left_mask], depth + 1), _build_tree(subset[~left_mask], depth + 1))

    def _path_length(point, tree, depth=0):
        if isinstance(tree, (int, float)):
            return depth + _c(tree)
        feat, split, left, right = tree
        if point[feat] < split:
            return _path_length(point, left, depth + 1)
        return _path_length(point, right, depth + 1)

    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, size=min(subsample, n), replace=False)
        trees.append(_build_tree(X[idx]))

    avg_paths = np.zeros(n)
    for i in range(n):
        avg_paths[i] = np.mean([_path_length(X[i], t) for t in trees])

    c_n = _c(min(subsample, n))
    scores = 2.0 ** (-avg_paths / max(c_n, 1e-10))

    threshold = float(np.quantile(scores, 1 - contamination))
    labels = (scores >= threshold).astype(int)

    return DescriptiveResult(
        name="isolation_anomaly",
        value=scores,
        extra={"labels": labels, "threshold": threshold, "n_anomalies": int(labels.sum()), "n": n, "p": p},
    )


anoiso = anomaly_isolation


def cheatsheet() -> str:
    return 'anomaly_isolation({}) -> Anomaly detection (isolation-forest style).'
