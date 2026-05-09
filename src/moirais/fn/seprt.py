# moirais.fn — function file (hadesllm/moirais)
"""Separation detection in logistic regression."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def detect_separation(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
) -> DescriptiveResult:
    """Detect complete or quasi-complete separation in logistic regression.

    Uses the linear-programming approach: checks whether there exists a
    vector :math:`\\beta` such that :math:`X\\beta \\ge 0` when y=1 and
    :math:`X\\beta \\le 0` when y=0 (complete separation), or nearly so
    (quasi-separation).

    Parameters
    ----------
    y : (n,) binary {0, 1}
    X : (n, p) predictors
    add_intercept : bool

    Returns
    -------
    DescriptiveResult
        ``value`` is True if separation detected.
        ``extra`` has ``type`` ('complete', 'quasi', or 'none'),
        ``separating_variable`` indices, and ``max_coef``.

    References
    ----------
    Albert, A. & Anderson, J. A. (1984). On the existence of maximum
    likelihood estimates in logistic regression models. *Biometrika*,
    71(1), 1--10.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    p = X.shape[1]

    sep_vars = []
    start = 1 if add_intercept else 0
    for j in range(start, p):
        vals_1 = X[y == 1, j]
        vals_0 = X[y == 0, j]
        if len(vals_1) == 0 or len(vals_0) == 0:
            continue
        if np.min(vals_1) > np.max(vals_0) or np.max(vals_1) < np.min(vals_0) or np.min(vals_1) >= np.max(vals_0) or np.max(vals_1) <= np.min(vals_0):
            sep_vars.append(j - (1 if add_intercept else 0))

    from scipy import special
    beta = np.zeros(p)
    for _ in range(200):
        eta = X @ beta
        mu = special.expit(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = mu * (1 - mu)
        z = eta + (y - mu) / W
        try:
            beta = np.linalg.solve((X * W[:, None]).T @ X + np.eye(p) * 1e-10,
                                   (X * W[:, None]).T @ z)
        except np.linalg.LinAlgError:
            break

    max_coef = float(np.max(np.abs(beta[start:])))
    complete_sep = len(sep_vars) > 0
    quasi_sep = max_coef > 10.0 and not complete_sep

    if complete_sep:
        sep_type = "complete"
    elif quasi_sep:
        sep_type = "quasi"
    else:
        sep_type = "none"

    return DescriptiveResult(
        name="SeparationDetection",
        value=complete_sep or quasi_sep,
        extra={
            "type": sep_type,
            "separating_variables": sep_vars,
            "max_abs_coefficient": max_coef,
            "n": n,
            "n_events": int(np.sum(y)),
        },
    )


seprt = detect_separation


def cheatsheet() -> str:
    return "detect_separation({}) -> Separation detection in logistic regression."
