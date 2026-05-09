"""Internal CFA engine using ML estimation (numpy/scipy only).

Fits confirmatory factor analysis models by minimizing the ML
discrepancy function (Joreskog, 1969). Used by all cfa*/mi* fn modules.

References
----------
Joreskog, K.G. (1969). A general approach to confirmatory maximum
    likelihood factor analysis. Psychometrika, 34(2), 183-202.
Hu, L. & Bentler, P.M. (1999). Cutoff criteria for fit indexes.
    Structural Equation Modeling, 6(1), 1-55.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _implied_cov(lam: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Model-implied covariance: Sigma = Lambda @ Phi @ Lambda' + Theta."""
    return lam @ phi @ lam.T + np.diag(theta)


def _ml_discrepancy(S: np.ndarray, Sigma: np.ndarray, p: int) -> float:
    """ML discrepancy: F = log|Sigma| + tr(S @ Sigma^-1) - log|S| - p."""
    try:
        sign_s, logdet_s = np.linalg.slogdet(S)
        sign_sig, logdet_sig = np.linalg.slogdet(Sigma)
        if sign_s <= 0 or sign_sig <= 0:
            return 1e10
        Sigma_inv = np.linalg.inv(Sigma)
        return float(logdet_sig + np.trace(S @ Sigma_inv) - logdet_s - p)
    except np.linalg.LinAlgError:
        return 1e10


def _pack_params(lam_free: np.ndarray, phi_free: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Pack free parameters into a 1-D vector."""
    # phi_free = lower-triangle of Phi (including diagonal)
    return np.concatenate([lam_free, phi_free, theta])


def _unpack_params(
    params: np.ndarray, n_lam: int, n_phi: int, p: int, n_factors: int, pattern: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack 1-D vector into Lambda, Phi, Theta."""
    lam_free = params[:n_lam]
    phi_free = params[n_lam : n_lam + n_phi]
    theta = np.abs(params[n_lam + n_phi :])  # residual variances > 0

    # Reconstruct Lambda
    lam = np.zeros((p, n_factors))
    lam[pattern > 0] = lam_free

    # Reconstruct Phi (correlation matrix of factors)
    phi = np.eye(n_factors)
    idx = 0
    for i in range(n_factors):
        for j in range(i + 1):
            if i == j:
                phi[i, j] = 1.0  # Fix diagonal
            else:
                phi[i, j] = np.clip(phi_free[idx], -0.999, 0.999)
                phi[j, i] = phi[i, j]
                idx += 1

    return lam, phi, theta


def fit_cfa(S: np.ndarray, n: int, structure: dict[str, list[int]], p: int, *, max_iter: int = 1000) -> dict:
    """Fit a CFA model via ML estimation.

    Parameters
    ----------
    S : ndarray (p x p)
        Sample covariance matrix.
    n : int
        Sample size.
    structure : dict
        Factor name -> list of 0-indexed item indices.
    p : int
        Number of observed variables.
    max_iter : int
        Maximum optimizer iterations.

    Returns
    -------
    dict with keys: cfi, tli, rmsea, srmr, chi2, df, p_value,
        loadings (dict), residuals (ndarray), implied_cov (ndarray),
        converged (bool).
    """
    factor_names = list(structure.keys())
    n_factors = len(factor_names)

    # Build pattern matrix (which loadings are free)
    pattern = np.zeros((p, n_factors), dtype=int)
    for fi, fname in enumerate(factor_names):
        for idx in structure[fname]:
            pattern[idx, fi] = 1
    n_lam = int(pattern.sum())

    # Number of free factor correlations (lower triangle, no diag)
    n_phi = n_factors * (n_factors - 1) // 2

    # Initial values
    lam_init = np.full(n_lam, 0.7)
    phi_init = np.full(n_phi, 0.3)
    theta_init = np.diag(S) * 0.5

    x0 = _pack_params(lam_init, phi_init, theta_init)

    def objective(params):
        lam, phi, theta = _unpack_params(params, n_lam, n_phi, p, n_factors, pattern)
        Sigma = _implied_cov(lam, phi, theta)
        return _ml_discrepancy(S, Sigma, p)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": max_iter})

    lam, phi, theta = _unpack_params(result.x, n_lam, n_phi, p, n_factors, pattern)
    Sigma = _implied_cov(lam, phi, theta)

    # F_ML at solution
    f_ml = _ml_discrepancy(S, Sigma, p)
    chi2 = max((n - 1) * f_ml, 0.0)

    # Degrees of freedom
    n_observed = p * (p + 1) // 2
    n_free = n_lam + n_phi + p  # loadings + correlations + residual variances
    df = max(n_observed - n_free, 1)

    # Null model (all items uncorrelated)
    S_null = np.diag(np.diag(S))
    f_null = _ml_discrepancy(S, S_null, p)
    chi2_null = max((n - 1) * f_null, 0.0)
    df_null = p * (p - 1) // 2

    # Fit indices
    cfi = _compute_cfi(chi2, df, chi2_null, df_null)
    tli = _compute_tli(chi2, df, chi2_null, df_null)
    rmsea = _compute_rmsea(chi2, df, n)
    srmr = _compute_srmr(S, Sigma)
    aic = chi2 + 2 * n_free
    bic = chi2 + np.log(n) * n_free

    from scipy import stats as sp

    p_value = float(1 - sp.chi2.cdf(chi2, df)) if df > 0 else 1.0

    # Loadings as dict
    loadings = {}
    for fi, fname in enumerate(factor_names):
        loadings[fname] = {}
        for idx in structure[fname]:
            loadings[fname][idx] = float(lam[idx, fi])

    return {
        "cfi": float(np.clip(cfi, 0, 1)),
        "tli": float(np.clip(tli, -1, 1)),
        "rmsea": float(max(rmsea, 0)),
        "srmr": float(srmr),
        "chi2": float(chi2),
        "df": int(df),
        "p_value": float(p_value),
        "aic": float(aic),
        "bic": float(bic),
        "loadings": loadings,
        "residuals": S - Sigma,
        "implied_cov": Sigma,
        "converged": result.success,
        "lambda": lam,
        "phi": phi,
        "theta": theta,
        "f_ml": float(f_ml),
        "chi2_null": float(chi2_null),
        "df_null": int(df_null),
    }


def _compute_cfi(chi2, df, chi2_null, df_null):
    d_model = max(chi2 - df, 0)
    d_null = max(chi2_null - df_null, 0)
    if d_null < 1e-10:
        return 1.0
    return 1 - d_model / d_null


def _compute_tli(chi2, df, chi2_null, df_null):
    if df_null < 1 or df < 1:
        return 1.0
    ratio_null = chi2_null / df_null
    ratio_model = chi2 / df
    denom = ratio_null - 1
    if abs(denom) < 1e-10:
        return 1.0
    return (ratio_null - ratio_model) / denom


def _compute_rmsea(chi2, df, n):
    if df < 1:
        return 0.0
    val = (chi2 / df - 1) / (n - 1)
    return float(np.sqrt(max(val, 0)))


def _compute_srmr(S, Sigma):
    p = S.shape[0]
    diff = S - Sigma
    # Standardize
    d = np.sqrt(np.diag(S))
    d[d < 1e-10] = 1.0
    D = np.outer(d, d)
    r = diff / D
    mask = np.tril(np.ones((p, p), dtype=bool))
    return float(np.sqrt(np.mean(r[mask] ** 2)))


def structure_to_indices(structure: dict[str, list[str]], item_names: list[str]) -> dict[str, list[int]]:
    """Convert factor -> item-name structure to factor -> index structure."""
    name_to_idx = {name: i for i, name in enumerate(item_names)}
    return {fname: [name_to_idx[item] for item in items if item in name_to_idx] for fname, items in structure.items()}


def get_mapq_structure(items: list[str] | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """Get MAPQ 4-factor structure and item list.

    Parameters
    ----------
    items : list of str or None
        Override item names. If None, uses MAPQ defaults.

    Returns
    -------
    tuple of (structure dict, item_names list)
    """
    from moirais.fn._mapq_const import ALL_ITEMS, SUBSCALES

    if items is not None:
        item_names = list(items)
    else:
        item_names = list(ALL_ITEMS)
    structure = {k: [i for i in v if i in item_names] for k, v in SUBSCALES.items()}
    return structure, item_names


def cov_from_data(data, item_names: list[str]) -> tuple[np.ndarray, int]:
    """Extract covariance matrix and sample size from data."""
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        X = data[item_names].to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)
    # Drop rows with NaN
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n = X.shape[0]
    S = np.cov(X, rowvar=False, ddof=1)
    return S, n
