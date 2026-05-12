# morie.fn -- function file (hadesllm/morie)
"""Regularized discriminant analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.5,
    gamma: float = 0.0,
) -> DescriptiveResult:
    """Regularized Discriminant Analysis (Friedman 1989).

    Interpolates between LDA (alpha=0) and QDA (alpha=1), with additional
    shrinkage toward a scaled identity (gamma).

    Parameters
    ----------
    X_train : ndarray (n, p)
        Training features.
    y_train : ndarray (n,)
        Training labels.
    X_test : ndarray (m, p)
        Test features.
    alpha : float
        Interpolation between pooled (0) and class-specific (1) covariance.
    gamma : float
        Shrinkage toward scaled identity (0 = no shrinkage).

    Returns
    -------
    DescriptiveResult
        ``value`` is predicted labels. ``extra`` has ``log_posteriors``.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)
    Xt = np.asarray(X_test, dtype=np.float64)
    n, p = X.shape
    classes = np.unique(y)

    means = {}
    covs = {}
    priors = {}

    Sw = np.zeros((p, p))
    for c in classes:
        Xc = X[y == c]
        means[c] = Xc.mean(axis=0)
        covs[c] = np.cov(Xc, rowvar=False, ddof=1)
        priors[c] = Xc.shape[0] / n
        Sw += (Xc.shape[0] - 1) * covs[c]
    Sw /= (n - len(classes))

    m = Xt.shape[0]
    log_posts = np.zeros((m, len(classes)))

    for j, c in enumerate(classes):
        Sigma_c = alpha * covs[c] + (1 - alpha) * Sw
        trace_avg = np.trace(Sigma_c) / p
        Sigma_c = (1 - gamma) * Sigma_c + gamma * trace_avg * np.eye(p)
        Sigma_c += np.eye(p) * 1e-8

        cov_inv = np.linalg.inv(Sigma_c)
        _, logdet = np.linalg.slogdet(Sigma_c)
        diff = Xt - means[c]
        log_posts[:, j] = (
            -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            - 0.5 * logdet
            + np.log(priors[c])
        )

    preds = classes[np.argmax(log_posts, axis=1)]

    return DescriptiveResult(
        name="RDA",
        value=preds,
        extra={"log_posteriors": log_posts, "alpha": alpha, "gamma": gamma},
    )


rdanl = rda


def cheatsheet() -> str:
    return "rda({}) -> Regularized discriminant analysis (Friedman 1989)."
