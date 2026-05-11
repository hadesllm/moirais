# morie.fn — function file (hadesllm/morie)
"""DR-learner (doubly-robust) for CATE estimation.

The DR-learner uses augmented IPW pseudo-outcomes as a robust signal,
then regresses these on covariates for heterogeneous effect estimates.

References
----------
Kennedy, E. H. (2023). Towards optimal doubly robust estimation of
heterogeneous causal effects. *Electronic Journal of Statistics*,
17(2), 3008-3049.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., & Robins, J. (2018). Double/debiased machine learning.
*The Econometrics Journal*, 21(1), C1-C68.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm as _norm

__all__ = ["drlea"]


def drlea(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate CATE via the DR-learner (doubly-robust meta-learner).

    Constructs the augmented IPW (AIPW) pseudo-outcome:

    .. math::

        \tilde{Y}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
            + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{g}(X_i)}
            - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{g}(X_i)}

    then regresses :math:`\tilde{Y}` on :math:`X` to obtain
    :math:`\hat{\tau}(x)`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    ps_trim : float
        Propensity score trimming threshold ``[ps_trim, 1-ps_trim]``.
    alpha : float
        Significance level for ATE confidence interval.

    Returns
    -------
    dict
        ``cate``, ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pseudo_outcome``, ``n``, ``method``.

    References
    ----------
    Kennedy (2023). Electronic Journal of Statistics, 17(2), 3008-3049.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)
    if not (len(T) == n == X.shape[0]):
        raise ValueError("Y, T, X must have the same number of observations.")

    Xa = np.column_stack([np.ones(n), X])

    # Propensity score model
    ps = np.clip(_irls(Xa, T), ps_trim, 1.0 - ps_trim)

    # Outcome models (T-learner)
    idx1, idx0 = T == 1, T == 0
    beta1 = np.linalg.lstsq(Xa[idx1], Y[idx1], rcond=None)[0]
    beta0 = np.linalg.lstsq(Xa[idx0], Y[idx0], rcond=None)[0]
    mu1 = Xa @ beta1
    mu0 = Xa @ beta0

    # AIPW pseudo-outcome
    pseudo = (
        mu1 - mu0
        + T * (Y - mu1) / ps
        - (1.0 - T) * (Y - mu0) / (1.0 - ps)
    )

    # Stage 2: regress pseudo-outcome on X for CATE
    gamma = np.linalg.lstsq(Xa, pseudo, rcond=None)[0]
    cate = Xa @ gamma

    ate = float(np.mean(pseudo))          # unbiased ATE from pseudo-outcomes
    se_ic = pseudo - ate                  # influence function residuals
    se = float(np.std(se_ic, ddof=1) / np.sqrt(n))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "cate": cate,
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "pseudo_outcome": pseudo,
        "n": n,
        "method": "DR-learner",
    }


def _irls(X: np.ndarray, y: np.ndarray, max_iter: int = 25) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        A = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        b = (X.T * W) @ (X @ beta + (y - p) / W)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
    return expit(X @ beta)


def cheatsheet() -> str:
    return "drlea(Y, T, X) -> DR-learner CATE (Kennedy 2023, EJS)."
