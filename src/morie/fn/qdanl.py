# morie.fn -- function file (hadesllm/morie)
"""Quadratic discriminant analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> DescriptiveResult:
    """Quadratic Discriminant Analysis classifier.

    Each class has its own covariance matrix (no pooling).

    Parameters
    ----------
    X_train : ndarray (n, p)
        Training feature matrix.
    y_train : ndarray (n,)
        Training class labels.
    X_test : ndarray (m, p)
        Test feature matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is predicted labels for X_test.
        ``extra`` has ``log_posteriors``, ``means``, ``priors``.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)
    Xt = np.asarray(X_test, dtype=np.float64)
    classes = np.unique(y)
    n = X.shape[0]

    means = {}
    covs = {}
    priors = {}

    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(axis=0)
        covs[c] = np.cov(Xc, rowvar=False, ddof=1) + np.eye(X.shape[1]) * 1e-6
        priors[c] = Xc.shape[0] / n

    m = Xt.shape[0]
    log_posts = np.zeros((m, len(classes)))

    for j, c in enumerate(classes):
        diff = Xt - means[c]
        cov_inv = np.linalg.inv(covs[c])
        _, logdet = np.linalg.slogdet(covs[c])
        log_posts[:, j] = (
            -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            - 0.5 * logdet
            + np.log(priors[c])
        )

    preds = classes[np.argmax(log_posts, axis=1)]

    return DescriptiveResult(
        name="QDA",
        value=preds,
        extra={
            "log_posteriors": log_posts,
            "means": {str(c): means[c].tolist() for c in classes},
            "priors": {str(c): float(priors[c]) for c in classes},
        },
    )


qdanl = qda


def cheatsheet() -> str:
    return "qda({}) -> Quadratic discriminant analysis."
