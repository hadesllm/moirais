# morie.fn — function file (hadesllm/morie)
"""Work program participation rates in custody."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import CrimeResult


def custody_work_program(
    df: pd.DataFrame,
    *,
    program_col: str = "work_program",
    id_col: str = "person_id",
) -> CrimeResult:
    """Work program participation rate.

    Parameters
    ----------
    df : DataFrame
    program_col : str
        Binary indicator (0/1) or categorical program column.
    id_col : str

    Returns
    -------
    CrimeResult
    """
    n = df[id_col].nunique() if id_col in df.columns else len(df)
    if df[program_col].dtype in ["int64", "float64", "bool"]:
        participating = int(df[program_col].sum())
    else:
        participating = int((df[program_col].notna() & (df[program_col] != "")).sum())
    rate = participating / max(n, 1)
    return CrimeResult(name="custody_work_program", rate=float(rate), n=participating, population=n)


cstwk = custody_work_program


def cheatsheet() -> str:
    return "custody_work_program({}) -> Work program participation rates in custody."
