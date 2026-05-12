# morie.fn -- function file (hadesllm/morie)
"""One-step estimator with Newton-Raphson influence-function correction.

The one-step estimator starts from an initial (possibly inefficient)
estimator and applies a single Newton-Raphson correction using the
efficient influence function to achieve semiparametric efficiency.

References
----------
Bickel, P. J., Klaassen, C. A. J., Ritov, Y., & Wellner, J. A. (1993).
*Efficient and Adaptive Estimation for Semiparametric Models*. Springer.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 13--14.

van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge
University Press. Chapter 25.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def onest(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    initial_estimate: float | None = None,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute the one-step semiparametric ATE estimator.

    Given an initial plug-in ATE estimate :math:`\hat{\psi}_0`, the
    one-step correction is:

    .. math::

        \hat{\psi}_{\text{os}} = \hat{\psi}_0
            + \frac{1}{n} \sum_{i=1}^{n} \mathrm{IF}(O_i; \hat{\psi}_0)

    where the efficient influence function for the ATE is:

    .. math::

        \mathrm{IF}(O; \psi) =
            \frac{T(Y - \hat{\mu}_1(X))}{\hat{e}(X)}
            - \frac{(1-T)(Y - \hat{\mu}_0(X))}{1 - \hat{e}(X)}
            + \hat{\mu}_1(X) - \hat{\mu}_0(X) - \psi

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    initial_estimate : float | None
        Initial plug-in ATE.  If *None*, uses simple difference in means.
    ps_trim : float
        Propensity scores clipped to ``[ps_trim, 1 - ps_trim]``.
    alpha : float
        Significance level.

    Returns
    -------
    dict[str, Any]
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``, ``initial_ate``,
        ``correction``, ``n``, ``method``.
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

    X1 = np.column_stack([T[:, None], X, np.ones(n)])
    beta = np.linalg.lstsq(X1, Y, rcond=None)[0]
    mu1 = np.column_stack([np.ones((n, 1)), X, np.ones(n)]) @ beta
    mu0 = np.column_stack([np.zeros((n, 1)), X, np.ones(n)]) @ beta

    if initial_estimate is None:
        initial_estimate = float(np.mean(mu1 - mu0))

    ic = (
        T * (Y - mu1) / ps
        - (1 - T) * (Y - mu0) / (1 - ps)
        + mu1 - mu0
        - initial_estimate
    )

    correction = float(np.mean(ic))
    ate = initial_estimate + correction

    ic_centered = ic - np.mean(ic)
    se = float(np.std(ic_centered, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "initial_ate": initial_estimate,
        "correction": correction,
        "n": n,
        "method": "OneStep",
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


onest_fn = onest


def cheatsheet() -> str:
    return "onest(Y, T, X) -> One-step ATE estimator (Kosorok 2008, Ch. 13-14)."
