# moirais.fn — function file (hadesllm/moirais)
"""Overall and by-group compliance rate."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def compliance_rate(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    group_col: str | None = None,
) -> dict | pd.DataFrame:
    """Compliance rate — proportion where flag_col == 1.

    Parameters
    ----------
    df : DataFrame
        Records with a binary compliance flag.
    flag_col : str
        Binary column (1 = compliant).
    group_col : str, optional
        If provided, return rates by group as a DataFrame.

    Returns
    -------
    dict or DataFrame
        If ``group_col`` is None: dict with ``n``, ``n_compliant``, ``rate``.
        Otherwise: DataFrame with ``[group_col, 'n', 'n_compliant', 'rate']``.
    """
    if group_col is None:
        n = len(df)
        nc = int(df[flag_col].sum())
        return RichResult(payload={"n": n, "n_compliant": nc, "rate": nc / n if n > 0 else 0.0})
    grp = df.groupby(group_col)[flag_col].agg(n="count", n_compliant="sum").reset_index()
    grp["rate"] = grp["n_compliant"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "compliance_rate({}) -> Overall and by-group compliance rate."
