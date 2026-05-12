"""Decision tree split via Gini / entropy."""
import numpy as np

from ._richresult import RichResult

__all__ = ["decision_tree_split"]


def decision_tree_split(x, y, *, criterion="gini", max_depth=None, seed=0):
    """Fit a decision tree classifier (CART) and return the split structure.

    Wraps sklearn.tree.DecisionTreeClassifier.  Reports the root impurity
    G(t) = 1 - sum p_k^2 (or H(t) = -sum p_k log p_k for "entropy"),
    the feature used at the root, and feature importances.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    criterion : "gini" | "entropy".
    max_depth : int or None.
    seed : int
        random_state for tie-breaking.

    Returns
    -------
    RichResult with payload: estimate (train accuracy), root_feature,
    root_threshold, root_impurity, n_leaves, feature_importances, n, method.
    """
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                 random_state=seed)
    clf.fit(X, y)
    tree = clf.tree_
    root_feat = int(tree.feature[0])
    root_thr = float(tree.threshold[0])
    root_imp = float(tree.impurity[0])
    acc = float(clf.score(X, y))
    return RichResult(payload={
        "estimate": acc,
        "train_accuracy": acc,
        "root_feature": root_feat,
        "root_threshold": root_thr,
        "root_impurity": root_imp,
        "n_leaves": int(clf.get_n_leaves()),
        "feature_importances": clf.feature_importances_.tolist(),
        "criterion": criterion,
        "n": int(n),
        "method": f"Decision tree (CART, {criterion})",
    })


def cheatsheet():
    return "dtrsp: CART decision tree (Gini/entropy)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    r = decision_tree_split(X, y, criterion="gini", max_depth=3)
    print("root feature:", r.root_feature, "  threshold:", r.root_threshold)
    print("root impurity:", r.root_impurity)
    print("n_leaves:", r.n_leaves, "  train accuracy:", r.train_accuracy)
    print("feature importances:", r.feature_importances)
