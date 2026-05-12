"""Linear SVM classifier via hinge loss gradient descent."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def svm_linear(X_train, y_train, X_test, C=1.0, lr=0.001, n_iter=1000, **kwargs) -> DescriptiveResult:
    r"""Linear SVM trained with sub-gradient descent on hinge loss.

    .. math::

        \\min_{\\mathbf{w},b} \\;
        \\frac{1}{2}\\|\\mathbf{w}\\|^2
        + C \\sum_{i=1}^{N} \\max(0,\\; 1 - y_i(\\mathbf{w}^\\top \\mathbf{x}_i + b))

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
        Binary labels (converted to {-1, +1}).
    X_test : array-like of shape (n_test, p)
    C : float
        Regularization parameter (default 1.0).
    lr : float
        Learning rate (default 0.001).
    n_iter : int
        Number of gradient descent iterations (default 1000).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Cortes, C. & Vapnik, V. (1995). Support-vector networks.
        *Machine Learning*, 20(3), 273--297.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_enc = np.where(y == classes[0], -1.0, 1.0)

    n, p = X_tr.shape
    w = np.zeros(p)
    b = 0.0

    for _ in range(n_iter):
        margin = y_enc * (X_tr @ w + b)
        misclassified = margin < 1
        dw = w - C * (y_enc[misclassified][:, None] * X_tr[misclassified]).sum(axis=0)
        db = -C * y_enc[misclassified].sum()
        w -= lr * dw
        b -= lr * db

    scores = X_te @ w + b
    preds_enc = np.sign(scores)
    preds_enc[preds_enc == 0] = 1.0
    predictions = np.where(preds_enc == -1.0, classes[0], classes[1])

    train_scores = X_tr @ w + b
    train_preds = np.sign(train_scores)
    train_preds[train_preds == 0] = 1.0
    train_acc = float(np.mean(train_preds == y_enc))

    return DescriptiveResult(
        name="svm_linear",
        value=train_acc,
        extra={
            "predictions": predictions,
            "decision_values": scores,
            "weights": w,
            "bias": b,
            "train_accuracy": train_acc,
            "C": C,
            "n_train": n,
            "n_test": len(X_te),
        },
    )


svmln = svm_linear


def cheatsheet() -> str:
    return "svm_linear({}) -> Linear SVM classifier via hinge loss gradient descent."
