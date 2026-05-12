"""Grid search with cross-validation."""
import numpy as np

from ._richresult import RichResult

__all__ = ["grid_search_cv"]


def grid_search_cv(x, y, *, estimator=None, param_grid=None, cv=5,
                    scoring=None, seed=0, task="auto"):
    """Exhaustive grid search via sklearn.model_selection.GridSearchCV.

    best_params = argmin CV_error(params).  Defaults illustrate a
    classification ridge-logistic search over C if you don't pass an
    estimator + param_grid.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    estimator : sklearn estimator or None.
    param_grid : dict | None.
    cv : int.
    scoring : str | None.
    seed : int
        random_state for estimators that accept it.
    task : "auto" | "classification" | "regression".

    Returns
    -------
    RichResult with payload: estimate (best CV score), best_params,
    best_score, cv_results (list of dicts), n, method.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GridSearchCV

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    if task == "auto":
        task = "classification" if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}) else "regression"

    if estimator is None:
        if task == "classification":
            estimator = LogisticRegression(max_iter=1000, random_state=seed)
            param_grid = param_grid or {"C": [0.01, 0.1, 1.0, 10.0]}
        else:
            estimator = Ridge(random_state=seed)
            param_grid = param_grid or {"alpha": [0.01, 0.1, 1.0, 10.0]}

    gs = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring=scoring)
    gs.fit(X, y)
    return RichResult(payload={
        "estimate": float(gs.best_score_),
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv_results_params": [dict(p) for p in gs.cv_results_["params"]],
        "cv_results_mean_score": gs.cv_results_["mean_test_score"].tolist(),
        "task": task,
        "n": int(n),
        "method": "Grid search CV",
    })


def cheatsheet():
    return "gsrch: grid search with cross-validation"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    r = grid_search_cv(X, y)
    print("best params:", r.best_params)
    print("best CV score:", r.best_score)
    print("CV scores per setting:", r.cv_results_mean_score)
