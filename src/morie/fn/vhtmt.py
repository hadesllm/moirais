"""Heterotrait-Monotrait ratio for discriminant validity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validity_htmt(
    data: pd.DataFrame | np.ndarray,
    subscales: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Heterotrait-Monotrait (HTMT) ratio of correlations.

    HTMT < 0.85 (conservative) or < 0.90 (liberal) indicates adequate
    discriminant validity (Henseler et al., 2015).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (respondents x items).
    subscales : dict, optional
        Mapping of subscale name to column name list.

    Returns
    -------
    DataFrame
        HTMT matrix (lower triangle meaningful, diagonal = NaN).

    References
    ----------
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion
    for assessing discriminant validity in variance-based structural
    equation modeling. *JAMS*, 43(1), 115--135.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    if subscales is None:
        subscales = {"total": list(data.columns)}

    names = list(subscales.keys())
    k = len(names)
    R = np.corrcoef(np.asarray(data[[c for cols in subscales.values() for c in cols]], dtype=np.float64), rowvar=False)

    # Map column index within full correlation matrix
    col_idx: dict[str, list[int]] = {}
    offset = 0
    for name, cols in subscales.items():
        col_idx[name] = list(range(offset, offset + len(cols)))
        offset += len(cols)

    htmt = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            idx_i = col_idx[names[i]]
            idx_j = col_idx[names[j]]
            # Heterotrait-heteromethod: correlations between items of different factors
            het = []
            for ii in idx_i:
                for jj in idx_j:
                    het.append(abs(R[ii, jj]))
            mean_het = np.mean(het)

            # Monotrait-heteromethod: geometric mean of avg within-factor correlations
            mono_i = []
            for a in range(len(idx_i)):
                for b in range(a + 1, len(idx_i)):
                    mono_i.append(abs(R[idx_i[a], idx_i[b]]))
            mono_j = []
            for a in range(len(idx_j)):
                for b in range(a + 1, len(idx_j)):
                    mono_j.append(abs(R[idx_j[a], idx_j[b]]))

            avg_i = np.mean(mono_i) if mono_i else np.nan
            avg_j = np.mean(mono_j) if mono_j else np.nan

            denom = np.sqrt(avg_i * avg_j) if avg_i > 0 and avg_j > 0 else np.nan
            htmt[i, j] = mean_het / denom if denom and denom > 0 else np.nan
            htmt[j, i] = htmt[i, j]

    return pd.DataFrame(htmt, index=names, columns=names)


def cheatsheet() -> str:
    return "validity_htmt({}) -> Heterotrait-Monotrait ratio for discriminant validity."
