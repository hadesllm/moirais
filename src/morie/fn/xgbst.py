"""XGBoost regularized objective — xgboost if available, sklearn HistGB fallback."""
import numpy as np

from ._richresult import RichResult

__all__ = ["xgboost_objective"]


def xgboost_objective(x, y, *, n_estimators=100, learning_rate=0.1,
                       max_depth=3, reg_lambda=1.0, reg_alpha=0.0,
                       task="auto", seed=0,
                       deterministic_seed: int | None = None):
    """Boosted-trees with XGBoost's regularized objective.

    L = sum_i l(y_i, y_hat_i) + sum_k Omega(f_k),
    Omega(f) = gamma T + (1/2) lambda ||w||^2 (+ alpha ||w||_1).

    Uses xgboost.XGB{Classifier,Regressor} when installed; otherwise falls
    back to sklearn.ensemble.HistGradientBoosting{Classifier,Regressor},
    which uses the same second-order LS objective with L2 leaf-shrinkage.
    The fallback path is flagged in the result so callers can branch on it.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    n_estimators, learning_rate, max_depth : XGBoost / HistGB hyperparams.
    reg_lambda : float
        L2 leaf penalty.
    reg_alpha : float
        L1 leaf penalty (XGBoost only).
    task : "auto" | "classification" | "regression".
    seed : int
        random_state.
    deterministic_seed : int or None, optional
        If supplied, the backend ``random_state`` is derived from the
        SHA-keyed :func:`morie._det_rng.r_seed` so Py<->R streams agree
        for the canonical fixture.  When ``None`` (default), behaviour
        is unchanged: ``seed`` drives ``random_state`` directly.

    Returns
    -------
    RichResult with payload: estimate (train score), feature_importances,
    backend ("xgboost" or "sklearn_histgb"), task, n, method.
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if task == "auto":
        task = "classification" if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}) else "regression"

    if deterministic_seed is not None:
        from morie._det_rng import r_seed
        rs = r_seed("xgbst", deterministic_seed)
    else:
        rs = seed

    backend = "xgboost"
    try:
        import xgboost as xgb  # type: ignore[import-not-found]
        Cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        m = Cls(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            random_state=rs, verbosity=0,
            eval_metric="logloss" if task == "classification" else "rmse",
        )
        m.fit(X, y)
    except ImportError:
        backend = "sklearn_histgb"
        from sklearn.ensemble import (
            HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        )
        Cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        m = Cls(
            max_iter=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, l2_regularization=reg_lambda,
            random_state=rs,
        )
        m.fit(X, y)

    score = float(m.score(X, y))
    importances = (m.feature_importances_.tolist()
                   if hasattr(m, "feature_importances_") else None)
    return RichResult(payload={
        "estimate": score,
        "train_score": score,
        "feature_importances": importances,
        "backend": backend,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "task": task,
        "n": int(n),
        "method": f"XGBoost-style boosting ({backend}, {task})",
    })


def cheatsheet():
    return "xgbst: XGBoost regularized boosting (sklearn HistGB fallback)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    r = xgboost_objective(X, y, n_estimators=50, seed=0)
    print("backend:", r.backend, "  task:", r.task)
    print("train score:", r.train_score)
    print("feature importances:", r.feature_importances)
