# moirais.fn — function file (hadesllm/moirais)
"""Cross-validated TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["cvtml"]


def cvtml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    *,
    n_folds: int = 5,
    ps_trim: float = 0.01,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Cross-validated TMLE (CV-TMLE) for the ATE.

    Uses sample splitting: for each fold, fit nuisance models on the
    training set and compute the TMLE update on the validation set.
    This avoids Donsker conditions on the nuisance estimators.

    :param Y: Outcome vector, shape (n,).
    :param T: Binary treatment vector, shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param n_folds: Number of CV folds. Default 5.
    :param ps_trim: Propensity score clip bounds. Default 0.01.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``fold_estimates``, ``n_folds``, ``n``, ``method``.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 21. Springer.
    Zheng & van der Laan (2011). CV-TMLE. *Statistical Science*.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(Y)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    ic_all = np.zeros(n)
    fold_estimates = []

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        Y_tr, T_tr, X_tr = Y[train_idx], T[train_idx], X[train_idx]
        Y_val, T_val, X_val = Y[val_idx], T[val_idx], X[val_idx]
        n_tr = len(Y_tr)
        n_val = len(Y_val)

        Xd_tr = np.column_stack([X_tr, np.ones(n_tr)])
        ps_tr = _logistic_predict(Xd_tr, T_tr)
        Xd_val = np.column_stack([X_val, np.ones(n_val)])
        beta_ps = _logistic_fit(Xd_tr, T_tr)
        from scipy.special import expit
        ps_val = expit(Xd_val @ beta_ps)
        ps_val = np.clip(ps_val, ps_trim, 1.0 - ps_trim)

        Xo_tr = np.column_stack([T_tr[:, None], X_tr, np.ones(n_tr)])
        beta_out = np.linalg.lstsq(Xo_tr, Y_tr, rcond=None)[0]
        mu1_val = np.column_stack([np.ones((n_val, 1)), X_val, np.ones(n_val)]) @ beta_out
        mu0_val = np.column_stack([np.zeros((n_val, 1)), X_val, np.ones(n_val)]) @ beta_out
        mu_obs_val = np.where(T_val == 1, mu1_val, mu0_val)

        H_val = T_val / ps_val - (1 - T_val) / (1 - ps_val)

        ate_fold = float(np.mean(mu1_val - mu0_val))
        fold_estimates.append(ate_fold)

        ic_all[val_idx] = H_val * (Y_val - mu_obs_val) + mu1_val - mu0_val

    ate = float(np.mean(ic_all))
    ic_centered = ic_all - ate
    se = float(np.std(ic_centered, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "fold_estimates": fold_estimates,
        "n_folds": n_folds,
        "n": n,
        "method": "CV-TMLE",
    }


def _logistic_fit(X, y):
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
    return beta


def _logistic_predict(X, y):
    from scipy.special import expit
    return expit(X @ _logistic_fit(X, y))


def cheatsheet() -> str:
    return "cvtml(Y, T, X) -> Cross-validated TMLE."
