# moirais.fn — function file (hadesllm/moirais)
"""Clever covariate computation for TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["clvr"]


def clvr(
    T: np.ndarray,
    X: np.ndarray,
    *,
    ps_trim: float = 0.01,
) -> dict[str, Any]:
    r"""
    Compute the clever covariate for TMLE targeting the ATE.

    .. math::

        H(T, X) = \frac{T}{\hat{g}(X)} - \frac{1-T}{1 - \hat{g}(X)}

    For the treated: :math:`H_1 = 1/\hat{g}(X)`.
    For controls: :math:`H_0 = -1/(1-\hat{g}(X))`.

    :param T: Binary treatment vector, shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param ps_trim: Propensity score clip bounds. Default 0.01.
    :return: Dict with ``H`` (clever covariate), ``H1``, ``H0``,
        ``propensity_scores``, ``n``.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 16. Springer.
    van der Laan & Rose (2011). *Targeted Learning*. Springer.
    """
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(T)

    Xd = np.column_stack([X, np.ones(n)])
    ps = _logistic_predict(Xd, T)
    ps = np.clip(ps, ps_trim, 1.0 - ps_trim)

    H1 = 1.0 / ps
    H0 = -1.0 / (1.0 - ps)
    H = T * H1 + (1 - T) * H0

    return {
        "H": H,
        "H1": H1,
        "H0": H0,
        "propensity_scores": ps,
        "n": n,
    }


def _logistic_predict(X, y):
    from scipy.special import expit
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        p = expit(X @ beta)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        z = X @ beta + (y - p) / W
        try:
            beta = np.linalg.solve(X.T @ np.diag(W) @ X + 1e-8 * np.eye(X.shape[1]), X.T @ (W * z))
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "clvr(T, X) -> Clever covariate for TMLE."
