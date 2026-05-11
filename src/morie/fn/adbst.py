# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AdaBoost classifier with decision stumps."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Strike me down and I shall become more powerful than you can imagine."


def adaboost_bio(X_train, y_train, X_test, n_estimators=50, **kwargs) -> DescriptiveResult:
    """AdaBoost with decision stumps (Freund & Schapire, 1997).

    .. math::

        H(\\mathbf{x}) = \\text{sign}\\!\\left(
        \\sum_{t=1}^{T} \\alpha_t h_t(\\mathbf{x})\\right)

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    n_estimators : int
        Number of boosting rounds (default 50).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Freund, Y. & Schapire, R.E. (1997). A decision-theoretic
        generalization of on-line learning. *JCSS*, 55(1), 119--139.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_enc = np.where(y == classes[0], -1.0, 1.0)

    n, p = X_tr.shape
    w = np.ones(n) / n
    stumps = []
    alphas = []

    for _ in range(n_estimators):
        best_err = np.inf
        best_stump = None
        for j in range(p):
            thresholds = np.unique(X_tr[:, j])
            if len(thresholds) > 50:
                thresholds = np.percentile(X_tr[:, j], np.linspace(0, 100, 52)[1:-1])
            for t in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n)
                    if polarity == 1:
                        pred[X_tr[:, j] < t] = -1
                    else:
                        pred[X_tr[:, j] >= t] = -1
                    err = w[pred != y_enc].sum()
                    if err < best_err:
                        best_err = err
                        best_stump = (j, t, polarity, pred.copy())

        err = np.clip(best_err, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(-alpha * y_enc * best_stump[3])
        w /= w.sum()
        stumps.append(best_stump[:3])
        alphas.append(alpha)

    alphas = np.array(alphas)

    def _predict(X):
        n_s = len(X)
        agg = np.zeros(n_s)
        for (j, t, pol), a in zip(stumps, alphas):
            pred = np.ones(n_s)
            if pol == 1:
                pred[X[:, j] < t] = -1
            else:
                pred[X[:, j] >= t] = -1
            agg += a * pred
        signs = np.sign(agg)
        signs[signs == 0] = 1.0
        return np.where(signs == -1, classes[0], classes[1])

    predictions = _predict(X_te)
    train_acc = float(np.mean(_predict(X_tr) == y))

    return DescriptiveResult(
        name="adaboost_bio",
        value=train_acc,
        extra={
            "predictions": predictions,
            "train_accuracy": train_acc,
            "n_estimators": len(stumps),
            "alphas": alphas,
        },
    )


adbst = adaboost_bio


def cheatsheet() -> str:
    return "adaboost_bio({}) -> AdaBoost classifier with decision stumps."
