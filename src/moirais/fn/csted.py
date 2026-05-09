# moirais.fn — function file (hadesllm/moirais)
"""Education program enrollment and completion in custody."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def custody_education(
    df: pd.DataFrame,
    *,
    enrolled_col: str = "edu_enrolled",
    completed_col: str = "edu_completed",
) -> DescriptiveResult:
    """Education program enrollment and completion rates.

    Parameters
    ----------
    df : DataFrame
    enrolled_col : str
        Binary enrollment indicator.
    completed_col : str
        Binary completion indicator.

    Returns
    -------
    DescriptiveResult
    """
    n = len(df)
    n_enrolled = int(df[enrolled_col].sum())
    n_completed = int(df[completed_col].sum())
    enroll_rate = n_enrolled / max(n, 1)
    complete_rate = n_completed / max(n_enrolled, 1)
    return DescriptiveResult(
        name="custody_education",
        value=enroll_rate,
        extra={
            "enrollment_rate": enroll_rate,
            "completion_rate": complete_rate,
            "n_enrolled": n_enrolled,
            "n_completed": n_completed,
            "n": n,
        },
    )


csted = custody_education


def cheatsheet() -> str:
    return "custody_education({}) -> Education program enrollment and completion in custody."
