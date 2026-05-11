# morie.fn — function file (hadesllm/morie)
"""Doubly Robust Cross-fitted Targeted Estimator (DR-TMLE).

Combines TMLE targeting with sample-splitting (cross-fitting) to
eliminate Donsker conditions on nuisance estimators, allowing the use
of flexible ML models for both the outcome model and propensity score.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., & Robins, J. (2018). Double/debiased machine learning for
treatment and structural parameters. *The Econometrics Journal*,
21(1), C1--C68.

Zheng, W. & van der Laan, M. J. (2011). Cross-validated targeted
minimum-loss-based estimation. *International Journal of
Biostatistics*, 7(1), 1--20.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 14--15.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special, stats


def drctr(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    n_folds: int = 5,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    r"""Doubly robust cross-fitted ATE estimator (DR-TMLE).

    For each fold :math:`k`:

    1. Fit :math:`\hat{e}^{(-k)}(X)` and :math:`\hat{\mu}^{(-k)}(T,X)` on
       training folds.
    2. Compute the AIPW score on fold *k*:

    .. math::

        \hat{\psi}_k = \frac{1}{|I_k|} \sum_{i \in I_k}
            \left[
                \hat{\mu}_1^{(-k)}(X_i) - \hat{\mu}_0^{(-k)}(X_i)
                + \frac{T_i(Y_i - \hat{\mu}_1^{(-k)}(X_i))}{\hat{e}^{(-k)}(X_i)}
                - \frac{(1-T_i)(Y_i - \hat{\mu}_0^{(-k)}(X_i))}{1-\hat{e}^{(-k)}(X_i)}
            \right]

    3. Aggregate: :math:`\hat{\psi} = K^{-1}\sum_k \hat{\psi}_k`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Binary treatment (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariates, shape ``(n, p)``.
    n_folds : int
        Number of cross-fitting folds.
    ps_trim : float
        Propensity score clipping bounds.
    alpha : float
        Significance level.
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    dict[str, Any]
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``, ``fold_estimates``,
        ``n_folds``, ``n``, ``method``.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=n)

    ic_all = np.empty(n)
    fold_estimates = []

    for k in range(n_folds):
        test_mask = fold_ids == k
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            continue

        Y_tr, T_tr, X_tr = Y[train_mask], T[train_mask], X[train_mask]
        Y_te, T_te, X_te = Y[test_mask], T[test_mask], X[test_mask]
        n_tr = len(Y_tr)
        n_te = len(Y_te)

        Xd_tr = np.column_stack([X_tr, np.ones(n_tr)])
        Xd_te = np.column_stack([X_te, np.ones(n_te)])
        ps_beta = _logistic_fit(Xd_tr, T_tr)
        ps_te = special.expit(Xd_te @ ps_beta)
        ps_te = np.clip(ps_te, ps_trim, 1.0 - ps_trim)

        Xo_tr = np.column_stack([T_tr[:, None], X_tr, np.ones(n_tr)])
        beta_out = np.linalg.lstsq(Xo_tr, Y_tr, rcond=None)[0]
        mu1_te = np.column_stack([np.ones((n_te, 1)), X_te, np.ones(n_te)]) @ beta_out
        mu0_te = np.column_stack([np.zeros((n_te, 1)), X_te, np.ones(n_te)]) @ beta_out

        ic_k = (
            mu1_te - mu0_te
            + T_te * (Y_te - mu1_te) / ps_te
            - (1 - T_te) * (Y_te - mu0_te) / (1 - ps_te)
        )

        ic_all[test_mask] = ic_k
        fold_estimates.append(float(np.mean(ic_k)))

    ate = float(np.mean(ic_all))
    se = float(np.std(ic_all, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "fold_estimates": fold_estimates,
        "n_folds": n_folds,
        "n": n,
        "method": "DR-TMLE",
    }


def _logistic_fit(X, y):
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
    return beta


drctr_fn = drctr


def cheatsheet() -> str:
    return "drctr(Y, T, X) -> Doubly robust cross-fitted ATE (Kosorok 2008, Ch. 14-15)."
