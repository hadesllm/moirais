# moirais.fn — function file (hadesllm/moirais)
"""Confirmatory Factor Analysis (basic implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import CfaRes


def cfa(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str | int]],
) -> CfaRes:
    r"""Confirmatory Factor Analysis with approximate fit indices.

    Given a hypothesised factor structure, estimates loadings from the
    sample correlation matrix and computes CFI, TLI, RMSEA, and SRMR.

    The implied covariance is :math:`\Sigma = \Lambda \Phi \Lambda' + \Theta`
    where :math:`\Lambda` are loadings, :math:`\Phi` is the factor
    correlation, and :math:`\Theta` is the uniqueness diagonal.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    structure : dict
        ``{factor_name: [item_col1, item_col2, ...]}`` mapping.
        Items can be column names (DataFrame) or integer indices (ndarray).

    Returns
    -------
    CfaRes
        ``cfi``, ``tli``, ``rmsea``, ``srmr``, ``loadings``, ``residuals``.

    References
    ----------
    Hu, L. & Bentler, P. M. (1999). Cutoff criteria for fit indexes in
    covariance structure analysis. *Structural Equation Modeling*, 6(1), 1-55.
    DOI: 10.1080/10705519909540118
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape

    # Standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    S = np.cov(Z, rowvar=False)  # sample covariance (correlation of standardised)

    # Map item names to indices
    if isinstance(data, pd.DataFrame):
        col_map = {c: i for i, c in enumerate(data.columns)}
    else:
        col_map = {i: i for i in range(p)}

    n_factors = len(structure)
    factor_names = list(structure.keys())

    # Build Lambda (loadings matrix) via regression on factor scores
    Lambda = np.zeros((p, n_factors))
    loadings_dict: dict[str, dict[str, float]] = {}

    for f_idx, (fname, items) in enumerate(structure.items()):
        indices = [col_map[it] for it in items]
        # Factor score = mean of its items (unit-weighted)
        factor_score = Z[:, indices].mean(axis=1)
        # Regress each item on its factor
        f_var = np.var(factor_score, ddof=0)
        if f_var < 1e-12:
            f_var = 1.0
        loadings_dict[fname] = {}
        for idx in indices:
            lam = np.cov(Z[:, idx], factor_score)[0, 1] / f_var
            Lambda[idx, f_idx] = lam
            item_name = items[indices.index(idx)] if isinstance(items[0], str) else str(idx)
            loadings_dict[fname][str(item_name)] = float(lam)

    # Factor correlation
    factor_scores = np.zeros((n, n_factors))
    for f_idx, (fname, items) in enumerate(structure.items()):
        indices = [col_map[it] for it in items]
        factor_scores[:, f_idx] = Z[:, indices].mean(axis=1)
    Phi = np.corrcoef(factor_scores, rowvar=False)
    if Phi.ndim == 0:
        Phi = np.array([[1.0]])

    # Implied covariance
    Sigma_implied = Lambda @ Phi @ Lambda.T
    Theta = np.diag(np.maximum(np.diag(S) - np.diag(Sigma_implied), 1e-6))
    Sigma_implied += Theta

    # Residuals
    residuals = S - Sigma_implied

    # SRMR
    mask = np.tril_indices(p)
    s_vals = S[mask]
    r_vals = residuals[mask]
    sd_diag = np.sqrt(np.diag(S))
    sd_outer = np.outer(sd_diag, sd_diag)
    std_resid = residuals / np.where(sd_outer > 0, sd_outer, 1.0)
    srmr = float(np.sqrt(np.mean(std_resid[mask] ** 2)))

    # Chi-square-like fit statistic (ML discrepancy approximation)
    try:
        Sigma_inv = np.linalg.inv(Sigma_implied)
        log_det_sigma = np.linalg.slogdet(Sigma_implied)[1]
        log_det_s = np.linalg.slogdet(S)[1]
        F_ml = float(np.trace(S @ Sigma_inv) - p + log_det_sigma - log_det_s)
    except np.linalg.LinAlgError:
        F_ml = float(np.sum(residuals**2))
    F_ml = max(F_ml, 0.0)

    chi2_model = max((n - 1) * F_ml, 0.0)
    # Degrees of freedom
    n_params = int(np.sum(Lambda != 0)) + n_factors * (n_factors + 1) // 2
    df_model = max(p * (p + 1) // 2 - n_params, 1)

    # Null model (independence)
    chi2_null = max((n - 1) * (np.sum(np.log(np.maximum(np.diag(S), 1e-12))) - np.linalg.slogdet(S)[1]), 0.0)
    df_null = p * (p - 1) // 2

    # CFI
    numer = max(chi2_model - df_model, 0.0)
    denom = max(chi2_null - df_null, 0.0)
    cfi = 1.0 - numer / denom if denom > 0 else 1.0
    cfi = float(np.clip(cfi, 0.0, 1.0))

    # TLI (NNFI)
    ratio_m = chi2_model / df_model if df_model > 0 else 0.0
    ratio_n = chi2_null / df_null if df_null > 0 else 1.0
    tli = (ratio_n - ratio_m) / (ratio_n - 1.0) if ratio_n > 1.0 else 1.0
    tli = float(np.clip(tli, 0.0, 1.0))

    # RMSEA
    rmsea_val = max(chi2_model - df_model, 0.0) / (df_model * max(n - 1, 1))
    rmsea = float(np.sqrt(max(rmsea_val, 0.0)))

    return CfaRes(
        cfi=cfi,
        tli=tli,
        rmsea=rmsea,
        srmr=srmr,
        loadings=loadings_dict,
        residuals=residuals,
    )


def cheatsheet() -> str:
    return "cfa({}) -> Confirmatory Factor Analysis (basic implementation)."
