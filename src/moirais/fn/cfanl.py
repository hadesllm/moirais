# moirais.fn — function file (hadesllm/moirais)
"""Confirmatory factor analysis (ULS estimation)."""

from __future__ import annotations

import numpy as np

from ._containers import CfaRes


def cfa_uls(
    data: np.ndarray,
    model: dict[str, list[int]],
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CfaRes:
    """Confirmatory factor analysis via unweighted least squares.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    model : dict
        Factor specification: ``{factor_name: [item_indices]}``.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    CfaRes
        Fit indices (CFI, TLI, RMSEA, SRMR) and loadings.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    X = X - X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)

    n_factors = len(model)
    L = np.zeros((p, n_factors))
    for j, (_, items) in enumerate(model.items()):
        for i in items:
            L[i, j] = 0.5

    for _ in range(max_iter):
        Sigma = L @ L.T + np.diag(np.maximum(np.diag(S) - np.sum(L ** 2, axis=1), 1e-6))
        resid = S - Sigma

        grad = -2 * resid @ L
        L_new = L - 0.01 * grad

        for j, (_, items) in enumerate(model.items()):
            mask = np.ones(p, dtype=bool)
            mask[items] = False
            L_new[mask, j] = 0.0

        if np.max(np.abs(L_new - L)) < tol:
            L = L_new
            break
        L = L_new

    Sigma = L @ L.T + np.diag(np.maximum(np.diag(S) - np.sum(L ** 2, axis=1), 1e-6))
    resid = S - Sigma

    df_model = p * (p + 1) // 2 - (np.count_nonzero(L) + p)
    df_model = max(df_model, 1)

    f_model = np.sum(resid ** 2) / 2
    f_null = np.sum((S - np.diag(np.diag(S))) ** 2) / 2
    df_null = p * (p - 1) // 2

    chi_model = max((n - 1) * f_model, 0)
    chi_null = max((n - 1) * f_null, 0)

    cfi = 1.0 - max(chi_model - df_model, 0) / max(chi_null - df_null, 1e-12)
    cfi = np.clip(cfi, 0, 1)

    tli_num = chi_null / max(df_null, 1) - chi_model / max(df_model, 1)
    tli_den = chi_null / max(df_null, 1) - 1
    tli = tli_num / tli_den if abs(tli_den) > 1e-12 else 1.0
    tli = np.clip(tli, 0, 1)

    rmsea = np.sqrt(max(chi_model / df_model - 1, 0) / max(n - 1, 1))

    D = np.diag(1.0 / np.sqrt(np.diag(S)))
    S_std = D @ resid @ D
    srmr = np.sqrt(np.sum(np.tril(S_std) ** 2) / (p * (p + 1) / 2))

    loadings_dict = {}
    for j, (fname, items) in enumerate(model.items()):
        loadings_dict[fname] = {i: float(L[i, j]) for i in items}

    return CfaRes(
        cfi=float(cfi),
        tli=float(tli),
        rmsea=float(rmsea),
        srmr=float(srmr),
        loadings=loadings_dict,
        residuals=resid,
    )


cfanl = cfa_uls


def cheatsheet() -> str:
    return "cfa_uls({}) -> Confirmatory factor analysis (ULS estimation)."
