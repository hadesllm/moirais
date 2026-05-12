"""Learning curve — train/val error vs training-set size."""
import numpy as np

from ._richresult import RichResult

__all__ = ["learning_curve"]


def learning_curve(x, y, *, sizes=None, cv=5, seed=0, estimator=None):
    """Compute the learning curve for an estimator.

    Wraps sklearn.model_selection.learning_curve to produce mean train/val
    error curves across a sequence of training-set sizes.  The default
    estimator is LinearRegression; pass any sklearn-compatible estimator
    to override.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    sizes : array-like or None
        Fractions of training data, default np.linspace(0.1, 1.0, 5).
    cv : int
        CV folds.
    seed : int
        RNG seed (passed to estimator if it accepts random_state).
    estimator : sklearn estimator or None
        Defaults to LinearRegression().

    Returns
    -------
    RichResult with payload: estimate (val MSE at full size), train_sizes,
    train_scores (mean), val_scores (mean), n, method.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import learning_curve as _lc

    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if sizes is None:
        sizes = np.linspace(0.1, 1.0, 5)
    est = estimator if estimator is not None else LinearRegression()
    train_sizes, train_scores, val_scores = _lc(
        est, X, y, train_sizes=sizes, cv=cv,
        scoring="neg_mean_squared_error", random_state=seed, shuffle=True,
    )
    # Convert from "neg MSE" back to MSE
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    return RichResult(payload={
        "estimate": float(val_mse[-1]),
        "train_sizes": train_sizes.tolist(),
        "train_scores": train_mse.tolist(),
        "val_scores": val_mse.tolist(),
        "n": int(n),
        "method": "Learning curve (cv MSE)",
    })


def cheatsheet():
    return "lrcvg: learning curve (train/val error vs n)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -0.5, 2.0]) + rng.normal(scale=0.5, size=200)
    r = learning_curve(X, y, sizes=np.linspace(0.2, 1.0, 4), cv=3)
    print("train sizes:", r.train_sizes)
    print("train MSE:", r.train_scores)
    print("val MSE:", r.val_scores)
    print("final val MSE:", r.estimate)
