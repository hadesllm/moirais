# morie.fn — function file (hadesllm/morie)
"""Exploratory SEM (ESEM) via rotated CFA loadings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def cfa_esem(
    data: pd.DataFrame | np.ndarray,
    n_factors: int = 4,
    *,
    rotation: str = "varimax",
) -> DescriptiveResult:
    """Exploratory Structural Equation Modeling (ESEM).

    Extracts factors via eigendecomposition and applies rotation,
    then computes fit indices from reproduced correlation.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data.
    n_factors : int
        Number of factors (default 4).
    rotation : str
        Rotation method: "varimax" or "none" (default "varimax").

    Returns
    -------
    DescriptiveResult
        value=dict with loadings and fit indices.

    References
    ----------
    Asparouhov, T. & Muthen, B. (2009). Exploratory structural equation
    modeling. Structural Equation Modeling, 16(3), 397-438.
    """
    X = np.asarray(data, dtype=np.float64)
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
        X = data.dropna().to_numpy(dtype=np.float64)
    else:
        names = [f"item_{j}" for j in range(X.shape[1])]

    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(-evals)
    evals = evals[order]
    evecs = evecs[:, order]

    L = evecs[:, :n_factors] * np.sqrt(np.maximum(evals[:n_factors], 0))

    if rotation == "varimax" and n_factors > 1:
        L = _varimax(L)

    R_hat = L @ L.T + np.diag(1 - np.sum(L**2, axis=1))
    residuals = R - R_hat
    srmr = float(np.sqrt(np.mean(np.tril(residuals, -1) ** 2)))

    loadings = {}
    for f in range(n_factors):
        loadings[f"F{f + 1}"] = {names[i]: float(L[i, f]) for i in range(p)}

    communalities = np.sum(L**2, axis=1)

    return DescriptiveResult(
        name="ESEM",
        value={
            "loadings": loadings,
            "communalities": {names[i]: float(communalities[i]) for i in range(p)},
            "srmr": srmr,
        },
        extra={"n": n, "p": p, "n_factors": n_factors, "rotation": rotation},
    )


def _varimax(L, max_iter=100, tol=1e-6):
    """Varimax rotation."""
    p, k = L.shape
    T = np.eye(k)
    for _ in range(max_iter):
        B = L @ T
        u, s, vt = np.linalg.svd(L.T @ (B**3 - B @ np.diag(np.sum(B**2, axis=0)) / p))
        T_new = u @ vt
        if np.max(np.abs(T_new - T)) < tol:
            break
        T = T_new
    return L @ T


esem = cfa_esem


def cheatsheet() -> str:
    return "cfa_esem({}) -> Exploratory SEM (ESEM) via rotated CFA loadings."
