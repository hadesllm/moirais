"""Random Forest ensemble (bagging + feature subsampling)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["random_forest_ensemble"]


def random_forest_ensemble(x, y, *, n_estimators=100, max_depth=None,
                           task="auto", seed=0):
    """Random Forest via sklearn.ensemble.RandomForest{Classifier,Regressor}.

    f(x) = (1/B) sum_b f_b(x), bagged decision trees with sqrt(p) features
    sampled per split (classifier default).

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    n_estimators : int
        Number of trees.
    max_depth : int or None.
    task : "auto" | "classification" | "regression"
        "auto" infers from y dtype (integer-typed y -> classification).
    seed : int
        random_state for reproducibility.

    Returns
    -------
    RichResult with payload: estimate (train accuracy for classifier, R^2
    for regressor), oob_score (if computable), feature_importances, n, method.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    if task == "auto":
        task = "classification" if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}) else "regression"

    if task == "classification":
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=seed,
                                     oob_score=True, bootstrap=True)
        clf.fit(X, y)
        score = float(clf.score(X, y))
        oob = float(getattr(clf, "oob_score_", np.nan))
    else:
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=seed,
                                    oob_score=True, bootstrap=True)
        reg.fit(X, y)
        score = float(reg.score(X, y))
        oob = float(getattr(reg, "oob_score_", np.nan))
        clf = reg

    return RichResult(payload={
        "estimate": score,
        "train_score": score,
        "oob_score": oob,
        "feature_importances": clf.feature_importances_.tolist(),
        "n_estimators": int(n_estimators),
        "task": task,
        "n": int(n),
        "method": f"Random Forest ({task})",
    })


def cheatsheet():
    return "rfens: random forest (bagged trees + feature subsampling)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    r = random_forest_ensemble(X, y, n_estimators=50, seed=0)
    print("task:", r.task, "  train score:", r.train_score)
    print("oob score:", r.oob_score)
    print("feature importances:", r.feature_importances)
