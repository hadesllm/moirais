"""Convergent validity: AVE > 0.5 per factor (Fornell & Larcker, 1981)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validity_convergent(
    data: pd.DataFrame | np.ndarray,
    subscales: dict[str, list[str]] | None = None,
) -> dict:
    """Assess convergent validity via Average Variance Extracted.

    AVE >= 0.5 for each subscale indicates adequate convergent validity
    (Fornell & Larcker, 1981).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (respondents x items).
    subscales : dict, optional
        Mapping of subscale name to list of column names.  If *None* and
        *data* is a DataFrame, all columns are treated as one factor.

    Returns
    -------
    dict
        Per-subscale AVE, loadings, and a boolean ``adequate`` flag.

    References
    ----------
    Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation
    models with unobservable variables and measurement error. *Journal of
    Marketing Research*, 18(1), 39--50.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    if subscales is None:
        subscales = {"total": list(data.columns)}

    results: dict[str, dict] = {}
    for name, cols in subscales.items():
        X = np.asarray(data[cols], dtype=np.float64)
        if X.shape[1] < 2:
            results[name] = {"ave": np.nan, "adequate": False, "loadings": []}
            continue
        # First principal component loadings as proxy
        X_c = X - X.mean(axis=0)
        cov = np.cov(X_c, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        pc1 = eigvecs[:, idx[0]]
        # Correlate each item with the component score
        scores = X_c @ pc1
        loadings = np.array([np.corrcoef(X_c[:, j], scores)[0, 1] for j in range(X.shape[1])])
        ave_val = float(np.mean(loadings**2))
        results[name] = {
            "ave": ave_val,
            "adequate": ave_val >= 0.5,
            "loadings": {cols[j]: float(loadings[j]) for j in range(len(cols))},
        }

    return results


def cheatsheet() -> str:
    return "validity_convergent({}) -> Convergent validity: AVE > 0.5 per factor (Fornell & Larcker"
