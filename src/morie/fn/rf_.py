# morie.fn — function file (hadesllm/morie)
"""Random forest (bagged CART) for regression."""

import numpy as np

from ._containers import DescriptiveResult
from .cart import _build_tree, _predict_one


def random_forest_simple(
    X: np.ndarray, y: np.ndarray, n_trees: int = 50, max_depth: int = 5, seed: int = 42
) -> DescriptiveResult:
    """
    Simple random forest regression (bagged CART, pure numpy).

    Each tree is trained on a bootstrap sample with sqrt(p) random
    features considered at each split.

    :param X: (n, p) feature matrix.
    :param y: (n,) target.
    :param n_trees: Number of trees.
    :param max_depth: Maximum tree depth.
    :param seed: Random seed.
    :return: DescriptiveResult with predictions and R-squared.

    References
    ----------
    Breiman L (2001). Random forests. Machine Learning, 45(1), 5-32.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, n, replace=True)
        tree = _build_tree(X[idx], y[idx], 0, max_depth, 2)
        trees.append(tree)
    preds = np.zeros(n)
    for tree in trees:
        preds += np.array([_predict_one(tree, x) for x in X])
    preds /= n_trees
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="random_forest",
        value=r2,
        extra={"predictions": preds, "r_squared": r2, "n_trees": n_trees, "max_depth": max_depth, "n": n},
    )


rf_ = random_forest_simple


def cheatsheet() -> str:
    return "random_forest_simple({}) -> Random forest (bagged CART) for regression."
