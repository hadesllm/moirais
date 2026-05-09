"""Polynomial kernel SVM classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Knowing yourself is the beginning of all wisdom. — Aristotle"


def svm_poly(X_train, y_train, X_test, C=1.0, degree=3, lr=0.01, n_iter=500, **kwargs) -> DescriptiveResult:
    """Polynomial kernel SVM via dual coordinate ascent.

    .. math::

        K(\\mathbf{x}_i, \\mathbf{x}_j) =
        (\\mathbf{x}_i^\\top \\mathbf{x}_j + 1)^d

    Parameters
    ----------
    X_train, y_train, X_test : array-like
    C : float
        Regularization (default 1.0).
    degree : int
        Polynomial degree (default 3).

    Returns
    -------
    DescriptiveResult
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y = np.asarray(y_train).ravel()
    classes = np.unique(y)
    y_enc = np.where(y == classes[0], -1.0, 1.0)

    n = X_tr.shape[0]

    def _poly_kernel(A, B):
        return (A @ B.T + 1) ** degree

    K = _poly_kernel(X_tr, X_tr)
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

    K_test = _poly_kernel(X_tr, X_te)
    scores = (alpha * y_enc) @ K_test + b
    preds_enc = np.sign(scores)
    preds_enc[preds_enc == 0] = 1.0
    predictions = np.where(preds_enc == -1.0, classes[0], classes[1])

    train_acc = float(np.mean(np.sign((alpha * y_enc) @ K + b) == y_enc))

    return DescriptiveResult(
        name="svm_poly",
        value=train_acc,
        extra={
            "predictions": predictions,
            "decision_values": scores,
            "n_support_vectors": int(sv_mask.sum()),
            "degree": degree,
            "train_accuracy": train_acc,
        },
    )


svmpl = svm_poly


def cheatsheet() -> str:
    return "svm_poly({}) -> Polynomial kernel SVM classifier."
