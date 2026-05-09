# moirais.fn — function file (hadesllm/moirais)
"""Efficient score function for semiparametric models.

The efficient score is the residual from projecting the parametric score
onto the nuisance tangent space. It achieves the semiparametric
efficiency bound when used for estimation.

References
----------
Bickel, P. J., Klaassen, C. A. J., Ritov, Y., & Wellner, J. A. (1993).
*Efficient and Adaptive Estimation for Semiparametric Models*. Springer.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapter 14.

Tsiatis, A. A. (2006). *Semiparametric Theory and Missing Data*.
Springer. Chapter 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def effsc(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute the efficient score for the ATE parameter.

    The efficient score :math:`S_{\mathrm{eff}}` for the ATE in a
    semiparametric model with treatment :math:`T`, outcome :math:`Y`, and
    covariates :math:`X` is:

    .. math::

        S_{\mathrm{eff}}(O_i; \psi) =
            \frac{T_i}{\hat{e}(X_i)} \bigl(Y_i - \hat{\mu}_1(X_i)\bigr)
            - \frac{1-T_i}{1-\hat{e}(X_i)} \bigl(Y_i - \hat{\mu}_0(X_i)\bigr)
            + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) - \psi

    Solving :math:`\sum S_{\mathrm{eff}} = 0` yields the semiparametrically
    efficient estimator. The information bound is
    :math:`I_{\mathrm{eff}} = E[S_{\mathrm{eff}}^2]`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    ps_trim : float
        Propensity score clipping bounds.
    alpha : float
        Significance level.

    Returns
    -------
    dict[str, Any]
        ``scores`` (array), ``estimate`` (solution to score equation),
        ``se``, ``ci_lower``, ``ci_upper``, ``info_bound``, ``n``.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    Xd = np.column_stack([X, np.ones(n)])
    ps = _logistic_predict(Xd, T)
    ps = np.clip(ps, ps_trim, 1.0 - ps_trim)

    Xo = np.column_stack([T[:, None], X, np.ones(n)])
    beta = np.linalg.lstsq(Xo, Y, rcond=None)[0]
    mu1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)]) @ beta
    mu0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)]) @ beta

    psi_init = float(np.mean(mu1 - mu0))

    scores = (
        T * (Y - mu1) / ps
        - (1 - T) * (Y - mu0) / (1 - ps)
        + mu1 - mu0
        - psi_init
    )

    estimate = psi_init + float(np.mean(scores))

    info_bound = float(np.mean(scores**2))
    se = float(np.sqrt(info_bound / n))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "scores": scores,
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - z * se,
        "ci_upper": estimate + z * se,
        "info_bound": info_bound,
        "n": n,
        "method": "EfficientScore",
    }


def _logistic_predict(X, y):
    from scipy import special
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        p = special.expit(X @ beta)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        z = X @ beta + (y - p) / W
        try:
            beta = np.linalg.solve(
                X.T @ np.diag(W) @ X + 1e-8 * np.eye(X.shape[1]),
                X.T @ (W * z),
            )
        except np.linalg.LinAlgError:
            break
    return special.expit(X @ beta)


effsc_fn = effsc


def cheatsheet() -> str:
    return "effsc(Y, T, X) -> Efficient score for ATE (Kosorok 2008, Ch. 14)."
