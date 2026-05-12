# morie.fn -- function file (hadesllm/morie)
"""Influence function computation for semiparametric estimators.

Computes the empirical influence function (IF) for a given functional,
enabling variance estimation, diagnostics, and one-step corrections.

References
----------
Hampel, F. R. (1974). The influence curve and its role in robust
estimation. *Journal of the American Statistical Association*,
69(346), 383--393.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapter 14.

Tsiatis, A. A. (2006). *Semiparametric Theory and Missing Data*.
Springer. Chapter 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def ifunc(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    functional: str = "ate",
    ps_trim: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute the efficient influence function for a target functional.

    For the ATE functional :math:`\psi = E[Y(1)] - E[Y(0)]`:

    .. math::

        \mathrm{IF}(O_i; \psi) =
            \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)}
            - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}
            + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) - \psi

    The variance of the estimator is :math:`\mathrm{Var}(\mathrm{IF}) / n`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment vector (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    functional : str
        Target functional: ``"ate"`` (default), ``"att"``, or ``"mean"``.
    ps_trim : float
        Propensity score clipping bounds.
    alpha : float
        Significance level for IF-based CI.

    Returns
    -------
    dict[str, Any]
        ``influence_values`` (array), ``estimate``, ``se``, ``ci_lower``,
        ``ci_upper``, ``n``, ``functional``.
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

    if functional == "ate":
        psi = float(np.mean(mu1 - mu0))
        ic = (
            T * (Y - mu1) / ps
            - (1 - T) * (Y - mu0) / (1 - ps)
            + mu1 - mu0 - psi
        )
    elif functional == "att":
        p1 = float(np.mean(T))
        psi = float(np.mean(T * (Y - mu0)) / p1)
        ic = (
            T * (Y - mu0) / p1
            - (1 - T) * ps * (Y - mu0) / ((1 - ps) * p1)
            - psi
        )
    elif functional == "mean":
        psi = float(np.mean(Y))
        ic = Y - psi
    else:
        raise ValueError(f"Unknown functional: {functional!r}. Use 'ate', 'att', or 'mean'.")

    estimate = psi + float(np.mean(ic))
    se = float(np.std(ic, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "influence_values": ic,
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - z * se,
        "ci_upper": estimate + z * se,
        "n": n,
        "functional": functional,
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


ifunc_fn = ifunc


def cheatsheet() -> str:
    return "ifunc(Y, T, X) -> Efficient influence function (Kosorok 2008, Ch. 14)."
