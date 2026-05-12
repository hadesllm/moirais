# morie.fn -- function file (hadesllm/morie)
"""Repeat placement frequency distribution."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_frequency(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
) -> pd.DataFrame:
    """Count how many placements each individual has, then tabulate.

    Returns a frequency distribution: how many individuals had 1, 2, 3, ...
    placements.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data (one row per placement event).
    id_col : str
        Column with unique individual identifiers.

    Returns
    -------
    DataFrame
        Columns: ``n_placements`` (1, 2, 3, ...) and ``n_individuals``.
    """
    per_person = df.groupby(id_col).size().reset_index(name="n_placements")
    freq = per_person.groupby("n_placements").size().reset_index(name="n_individuals")
    return freq.sort_values("n_placements").reset_index(drop=True)


rpfrq = rplace_frequency


def cheatsheet() -> str:
    return "rplace_frequency({}) -> Repeat placement frequency distribution."
