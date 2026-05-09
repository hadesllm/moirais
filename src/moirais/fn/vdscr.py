"""Discriminant validity: Fornell-Larcker criterion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validity_discriminant(
    data: pd.DataFrame | np.ndarray,
    subscales: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Fornell-Larcker discriminant validity test.

    For each pair of subscales, sqrt(AVE) of each factor must exceed the
    inter-factor correlation.  The returned DataFrame has sqrt(AVE) on the
    diagonal and inter-factor correlations off-diagonal.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (respondents x items).
    subscales : dict, optional
        Mapping of subscale name to column name list.

    Returns
    -------
    DataFrame
        Fornell-Larcker matrix with sqrt(AVE) on diagonal.

    References
    ----------
    Fornell, C., & Larcker, D. F. (1981). *Journal of Marketing Research*,
    18(1), 39--50.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    if subscales is None:
        subscales = {"total": list(data.columns)}

    names = list(subscales.keys())
    k = len(names)

    # Compute composite scores (mean of items per subscale)
    composites = pd.DataFrame({name: data[cols].mean(axis=1) for name, cols in subscales.items()})

    # Correlation matrix of composites
    corr = composites.corr().values

    # AVE per subscale
    aves = np.zeros(k)
    for i, (name, cols) in enumerate(subscales.items()):
        X = np.asarray(data[cols], dtype=np.float64)
        if X.shape[1] < 2:
            aves[i] = np.nan
            continue
        X_c = X - X.mean(axis=0)
        cov = np.cov(X_c, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]
        scores = X_c @ pc1
        loadings = np.array([np.corrcoef(X_c[:, j], scores)[0, 1] for j in range(X.shape[1])])
        aves[i] = float(np.mean(loadings**2))

    # Build Fornell-Larcker matrix
    fl = corr.copy()
    for i in range(k):
        fl[i, i] = np.sqrt(aves[i])

    return pd.DataFrame(fl, index=names, columns=names)


def cheatsheet() -> str:
    return "validity_discriminant({}) -> Discriminant validity: Fornell-Larcker criterion."
