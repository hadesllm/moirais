"""Random search for hyperparameter optimisation (Bergstra & Bengio 2012)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["random_search_cv"]


def random_search_cv(x, y, *, estimator=None, param_distributions=None,
                      n_iter=20, cv=5, scoring=None, seed=0, task="auto",
                      deterministic_seed: int | None = None):
    """Random hyperparameter search via sklearn.model_selection.RandomizedSearchCV.

    Samples n_iter configurations from `param_distributions`, evaluates
    each by k-fold CV, returns the best.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    estimator : sklearn estimator or None.
    param_distributions : dict | None.
    n_iter : int
        Number of random draws.
    cv : int.
    scoring : str | None.
    seed : int
        random_state — controls both the search and the estimator.
    task : "auto" | "classification" | "regression".
    deterministic_seed : int or None, optional
        If supplied, the sklearn ``random_state`` is derived from the
        SHA-keyed :func:`morie._det_rng.r_seed` so Py<->R streams agree
        for the canonical fixture.  When ``None`` (default), behaviour
        is unchanged: ``seed`` drives ``random_state`` directly.

    Returns
    -------
    RichResult with payload: estimate (best CV score), best_params,
    best_score, sampled_params, sampled_scores, n, method.
    """
    from scipy.stats import loguniform
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import RandomizedSearchCV

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    if deterministic_seed is not None:
        from morie._det_rng import r_seed
        rs_seed = r_seed("rndsr", deterministic_seed)
    else:
        rs_seed = seed

    if task == "auto":
        task = "classification" if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}) else "regression"

    if estimator is None:
        if task == "classification":
            estimator = LogisticRegression(max_iter=1000, random_state=rs_seed)
            param_distributions = param_distributions or {"C": loguniform(1e-3, 1e2)}
        else:
            estimator = Ridge(random_state=rs_seed)
            param_distributions = param_distributions or {"alpha": loguniform(1e-3, 1e2)}

    rs = RandomizedSearchCV(estimator, param_distributions=param_distributions,
                             n_iter=n_iter, cv=cv, scoring=scoring,
                             random_state=rs_seed)
    rs.fit(X, y)
    return RichResult(payload={
        "estimate": float(rs.best_score_),
        "best_params": rs.best_params_,
        "best_score": float(rs.best_score_),
        "sampled_params": [dict(p) for p in rs.cv_results_["params"]],
        "sampled_scores": rs.cv_results_["mean_test_score"].tolist(),
        "n_iter": int(n_iter),
        "task": task,
        "n": int(n),
        "method": "Random search CV (Bergstra & Bengio 2012)",
    })


def cheatsheet():
    return "rndsr: random search hyperparameter optimisation"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    r = random_search_cv(X, y, n_iter=10, seed=0)
    print("best params:", r.best_params)
    print("best CV score:", r.best_score)
    print("number of samples:", r.n_iter)
