"""Gender-specific substance use rates."""

import pandas as pd

from ._containers import DescriptiveResult


def substance_by_gender(
    df: pd.DataFrame,
    use_col: str = "substance_use",
    gender_col: str = "gender",
) -> DescriptiveResult:
    """Compute gender-specific substance use rates.

    Parameters
    ----------
    df : DataFrame
    use_col : str
        Binary substance use indicator column.
    gender_col : str
        Gender/sex column.

    Returns
    -------
    DescriptiveResult
    """
    if use_col not in df.columns:
        raise ValueError(f"Column '{use_col}' not found")
    if gender_col not in df.columns:
        raise ValueError(f"Column '{gender_col}' not found")

    grouped = df.groupby(gender_col)[use_col]
    tbl = pd.DataFrame({"rate": grouped.mean(), "n_users": grouped.sum(), "n_total": grouped.count()})

    return DescriptiveResult(
        name="substance_by_gender",
        value=tbl,
        extra={"n_groups": len(tbl), "overall_rate": float(df[use_col].mean())},
    )


sugen = substance_by_gender


def cheatsheet() -> str:
    return "substance_by_gender({}) -> Gender-specific substance use rates."
