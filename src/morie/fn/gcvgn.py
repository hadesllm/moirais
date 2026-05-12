# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation for genomic-prediction accuracy."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["genomic_cross_validation"]


def genomic_cross_validation(x, y, K: int = 5, lam: float = 1.0, seed: int = 0):
    """K-fold CV accuracy of a ridge predictor on `(x, y)`.

    For each fold, fit ridge regression on the training set, predict the
    held-out set, then report the Pearson correlation r = cor(y, y_hat)
    pooled across folds (Montesinos Lopez Ch 2 standard).

    Parameters
    ----------
    x : array-like (n,) or (n, p) — predictor matrix (e.g. markers).
    y : array-like (n,)
    K : int, default 5
    lam : float, default 1.0. Ridge penalty inside each fold.
    seed : int

    Returns
    -------
    RichResult with payload keys estimate (pooled r), r_per_fold, mse,
    mspe, slope, n, K, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 2.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    y_hat = np.zeros(n)
    r_per_fold = []
    for k in range(K):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(K) if j != k])
        Xtr, ytr = X[train], y[train]
        Xte = X[test]
        mu = ytr.mean(); x_mu = Xtr.mean(axis=0)
        Xtr_c = Xtr - x_mu
        beta = np.linalg.solve(Xtr_c.T @ Xtr_c + lam * np.eye(p),
                                Xtr_c.T @ (ytr - mu))
        y_hat[test] = (Xte - x_mu) @ beta + mu
        if len(test) > 1 and np.std(y[test]) > 0 and np.std(y_hat[test]) > 0:
            r_per_fold.append(float(np.corrcoef(y[test], y_hat[test])[0, 1]))
        else:
            r_per_fold.append(float("nan"))
    r_pooled = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 else float("nan")
    mse = float(np.mean((y - y_hat) ** 2))
    mspe = mse
    # Regression slope of y_hat on y (calibration slope)
    if np.var(y_hat) > 0:
        slope = float(np.cov(y_hat, y, ddof=1)[0, 1] / np.var(y, ddof=1))
    else:
        slope = float("nan")
    return RichResult(
        title=f"{K}-fold cross-validation (ridge predictor)",
        summary_lines=[
            ("n", n),
            ("K", K),
            ("ridge lambda", lam),
            ("pooled r", r_pooled),
            ("mean r across folds", float(np.nanmean(r_per_fold))),
            ("MSE", mse),
            ("calibration slope", slope),
        ],
        payload={
            "estimate": r_pooled,
            "r_per_fold": np.asarray(r_per_fold),
            "y_hat": y_hat,
            "mse": mse,
            "mspe": mspe,
            "slope": slope,
            "n": n,
            "K": K,
            "method": "K-fold cross-validation (ridge)",
        },
    )


def cheatsheet():
    return "gcvgn: K-fold CV for genomic prediction"


# CANONICAL TEST
# np.random.seed(15); X = np.random.randn(50, 4); beta = np.array([1,-1,0.5,0])
# y = X @ beta + 0.3*np.random.randn(50)
# r = genomic_cross_validation(X, y, K=5, seed=15); pooled r > 0.5.
