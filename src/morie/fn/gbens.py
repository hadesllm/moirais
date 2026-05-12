"""Gradient Boosting ensemble (sequential additive model)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["gradient_boosting_ensemble"]


def gradient_boosting_ensemble(x, y, *, n_estimators=100, learning_rate=0.1,
                                max_depth=3, task="auto", seed=0,
                                deterministic_seed: int | None = None):
    """Gradient boosting via sklearn.ensemble.GradientBoosting{Classifier,Regressor}.

    F_m(x) = F_{m-1}(x) + nu * h_m(x), where h_m fits the negative gradient
    of the loss at the previous prediction.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    n_estimators : int
        Number of boosting iterations.
    learning_rate : float
        Shrinkage nu.
    max_depth : int
        Depth of each weak learner.
    task : "auto" | "classification" | "regression".
    seed : int
        random_state.
    deterministic_seed : int or None, optional
        If supplied, the sklearn ``random_state`` is derived from the
        SHA-keyed :func:`morie._det_rng.r_seed` so Py<->R streams agree
        for the canonical fixture.  When ``None`` (default), behaviour
        is unchanged: ``seed`` drives ``random_state`` directly.

    Returns
    -------
    RichResult with payload: estimate (train accuracy / R^2),
    feature_importances, n_estimators, learning_rate, n, method.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
    )

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    if deterministic_seed is not None:
        from morie._det_rng import r_seed
        rs = r_seed("gbens", deterministic_seed)
    else:
        rs = seed

    if task == "auto":
        task = "classification" if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}) else "regression"

    Cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
    m = Cls(n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=rs)
    m.fit(X, y)
    score = float(m.score(X, y))
    return RichResult(payload={
        "estimate": score,
        "train_score": score,
        "feature_importances": m.feature_importances_.tolist(),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "task": task,
        "n": int(n),
        "method": f"Gradient Boosting ({task})",
    })


def cheatsheet():
    return "gbens: gradient boosting (sequential trees)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
    r = gradient_boosting_ensemble(X, y, n_estimators=50, learning_rate=0.1,
                                    max_depth=3, seed=0)
    print("task:", r.task, "  train score:", r.train_score)
    print("feature importances:", r.feature_importances)
