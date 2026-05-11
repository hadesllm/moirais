# morie.fn — function file (hadesllm/morie)
"""Isolation forest (pure numpy). 'Fear leads to anger.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class _IsoNode:
    __slots__ = ("left", "right", "split_feat", "split_val", "size")

    def __init__(self) -> None:
        self.left: _IsoNode | None = None
        self.right: _IsoNode | None = None
        self.split_feat: int = 0
        self.split_val: float = 0.0
        self.size: int = 0


def _build_tree(X: np.ndarray, depth: int, max_depth: int, rng: np.random.Generator) -> _IsoNode:
    node = _IsoNode()
    n, p = X.shape
    node.size = n
    if depth >= max_depth or n <= 1:
        return node
    feat = rng.integers(0, p)
    lo, hi = float(X[:, feat].min()), float(X[:, feat].max())
    if lo == hi:
        return node
    val = rng.uniform(lo, hi)
    node.split_feat = feat
    node.split_val = val
    mask = X[:, feat] < val
    node.left = _build_tree(X[mask], depth + 1, max_depth, rng)
    node.right = _build_tree(X[~mask], depth + 1, max_depth, rng)
    return node


def _path_length(x: np.ndarray, node: _IsoNode, depth: int) -> float:
    if node.left is None or node.right is None:
        return depth + _c_factor(node.size)
    if x[node.split_feat] < node.split_val:
        return _path_length(x, node.left, depth + 1)
    return _path_length(x, node.right, depth + 1)


def isolation_forest(
    X: np.ndarray,
    n_trees: int = 100,
    sample_size: int = 256,
    seed: int = 42,
) -> DescriptiveResult:
    """Isolation Forest anomaly detection (pure NumPy).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    n_trees : int, default 100
    sample_size : int, default 256
    seed : int, default 42

    Returns
    -------
    DescriptiveResult
        ``extra`` has ``scores`` (higher = more anomalous) and ``threshold``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    ss = min(sample_size, n)
    max_depth = int(np.ceil(np.log2(max(ss, 2))))

    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, size=ss, replace=False) if n > ss else np.arange(n)
        trees.append(_build_tree(X[idx], 0, max_depth, rng))

    avg_path = np.zeros(n)
    for i in range(n):
        avg_path[i] = np.mean([_path_length(X[i], t, 0) for t in trees])

    c = _c_factor(ss)
    scores = 2.0 ** (-avg_path / c) if c > 0 else np.zeros(n)
    threshold = float(np.percentile(scores, 95))

    return DescriptiveResult(
        name="Isolation Forest",
        value=float(np.mean(scores)),
        extra={
            "scores": scores,
            "threshold": threshold,
            "n_trees": n_trees,
            "sample_size": ss,
            "n": n,
        },
    )


isof = isolation_forest


def cheatsheet() -> str:
    return "_c_factor({}) -> Isolation forest (pure numpy). 'Fear leads to anger.' -- Yod"
