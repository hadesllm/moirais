"""RBF kernel SVM classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience there is no such thing as luck."


def svm_rbf(X_train, y_train, X_test, C=1.0, gamma=None, lr=0.01, n_iter=500, **kwargs) -> DescriptiveResult:
    """RBF kernel SVM using simplified SMO-style coordinate ascent.

    .. math::

        K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp\\bigl(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2\\bigr)

    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
    y_train : array-like of shape (n_train,)
    X_test : array-like of shape (n_test, p)
    C : float
        Regularization (default 1.0).
    gamma : float or None
        RBF bandwidth. Default 1/p.
    lr : float
        Learning rate for dual ascent.
    n_iter : int
        Iterations.

    Returns
    -------
    DescriptiveResult
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_enc = np.where(y == classes[0], -1.0, 1.0)

    n, p = X_tr.shape
    if gamma is None:
        gamma = 1.0 / max(p, 1)

    def _rbf_kernel(A, B):
        sq_A = np.sum(A**2, axis=1, keepdims=True)
        sq_B = np.sum(B**2, axis=1, keepdims=True)
        dist = sq_A - 2 * A @ B.T + sq_B.T
        return np.exp(-gamma * np.maximum(dist, 0))

    K = _rbf_kernel(X_tr, X_tr)
    alpha = np.zeros(n)

    for _ in range(n_iter):
        margin = (alpha * y_enc) @ K
        for i in range(n):
            grad = 1 - y_enc[i] * margin[i]
            alpha[i] = np.clip(alpha[i] + lr * grad, 0, C)
        margin = (alpha * y_enc) @ K

    sv_mask = alpha > 1e-7
    if sv_mask.sum() == 0:
        sv_mask[0] = True
    b = float(np.mean(y_enc[sv_mask] - (alpha * y_enc) @ K[:, sv_mask]))

    K_test = _rbf_kernel(X_tr, X_te)
    scores = (alpha * y_enc) @ K_test + b
    preds_enc = np.sign(scores)
    preds_enc[preds_enc == 0] = 1.0
    predictions = np.where(preds_enc == -1.0, classes[0], classes[1])

    train_scores = (alpha * y_enc) @ K + b
    train_acc = float(np.mean(np.sign(train_scores) == y_enc))

    return DescriptiveResult(
        name="svm_rbf",
        value=train_acc,
        extra={
            "predictions": predictions,
            "decision_values": scores,
            "n_support_vectors": int(sv_mask.sum()),
            "alpha": alpha,
            "bias": b,
            "gamma": gamma,
            "train_accuracy": train_acc,
        },
    )


svmrb = svm_rbf


def cheatsheet() -> str:
    return "svm_rbf({}) -> RBF kernel SVM classifier."
