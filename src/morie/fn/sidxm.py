r"""Single-index model via maximum score estimator (Manski 1975).

Estimates the coefficient direction in a binary-response single-index
model :math:`P(Y=1 \\mid X) = F(X^\\top \\beta)` where :math:`F` is
unknown.  The maximum score estimator maximizes

.. math::

    S_n(\\beta) = n^{-1} \\sum_{i=1}^n
        \\bigl[Y_i - \\tfrac{1}{2}\\bigr] \\,
        \\operatorname{sign}(X_i^\\top \\beta)

over the unit sphere :math:`\\|\\beta\\| = 1`.

References
----------
Manski, C. F. (1975). Maximum score estimation of the stochastic
    utility model of choice. *Journal of Econometrics*, 3(3), 205--228.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sidxm(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    n_starts: int = 50,
    random_state: int = 42,
) -> dict[str, Any]:
    r"""Maximum score estimator for single-index binary response.

    Parameters
    ----------
    Y : np.ndarray
        Binary outcome (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    n_starts : int
        Number of random starting directions to try.
    random_state : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        ``beta`` (unit-norm coefficients), ``score`` (max score value),
        ``n``, ``p``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if len(Y) != n:
        raise ValueError(f"Y length {len(Y)} != X rows {n}.")

    rng = np.random.default_rng(random_state)
    centered = Y - 0.5

    best_score = -np.inf
    best_beta = np.zeros(p)

    for _ in range(n_starts):
        b = rng.standard_normal(p)
        b /= np.linalg.norm(b) + 1e-12
        score = np.mean(centered * np.sign(X @ b))
        if score > best_score:
            best_score = score
            best_beta = b.copy()

    if best_beta[0] < 0:
        best_beta = -best_beta

    return {
        "beta": best_beta,
        "score": float(best_score),
        "n": n,
        "p": p,
        "method": "MaximumScore",
    }


sidxm_fn = sidxm


def cheatsheet() -> str:
    return "sidxm(Y, X) -> Single-index maximum score estimator (Manski 1975)."
