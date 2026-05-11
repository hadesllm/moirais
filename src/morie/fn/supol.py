"""Polysubstance use co-occurrence analysis."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def polysubstance(
    df: pd.DataFrame,
    substance_cols: list[str] | None = None,
) -> DescriptiveResult:
    """Compute polysubstance co-occurrence matrix and counts.

    Parameters
    ----------
    df : DataFrame
        Binary columns (0/1) for each substance.
    substance_cols : list of str or None
        Columns to analyse. If None, all columns are used.

    Returns
    -------
    DescriptiveResult
        value = co-occurrence DataFrame; extra has n_poly counts.
    """
    cols = substance_cols or list(df.columns)
    sub = df[cols].astype(float)
    cooccur = sub.T.dot(sub)
    vals = cooccur.values.copy()
    vals[np.diag_indices_from(vals)] = 0
    cooccur = pd.DataFrame(vals, index=cooccur.index, columns=cooccur.columns)

    n_substances_used = sub.sum(axis=1)
    poly_counts = {
        "mono_use": int((n_substances_used == 1).sum()),
        "poly_2": int((n_substances_used == 2).sum()),
        "poly_3plus": int((n_substances_used >= 3).sum()),
        "none": int((n_substances_used == 0).sum()),
    }

    return DescriptiveResult(
        name="polysubstance",
        value=cooccur,
        extra={"poly_counts": poly_counts, "n": len(df)},
    )


supol = polysubstance


def cheatsheet() -> str:
    return "polysubstance({}) -> Polysubstance use co-occurrence analysis."
