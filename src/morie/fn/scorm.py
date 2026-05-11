# morie.fn — function file (hadesllm/morie)
"""Covariance matrix."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"


def covariance_matrix(X, **kwargs) -> DescriptiveResult:
    """Compute the sample covariance matrix of *X*.

    Parameters
    ----------
    X : array-like, shape (N, p) or (N,)
        Data matrix with N observations and p variables.
        If 1-D, treats as a single variable.

    Returns
    -------
    DescriptiveResult
        ``value`` is the covariance matrix (ndarray).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cov_mat = np.cov(X, rowvar=False, ddof=1)
    return DescriptiveResult(
        name="covariance_matrix",
        value=cov_mat,
        extra={"shape": cov_mat.shape, "n": X.shape[0]},
    )


scorm = covariance_matrix


def cheatsheet() -> str:
    return "covariance_matrix({}) -> Covariance matrix."
