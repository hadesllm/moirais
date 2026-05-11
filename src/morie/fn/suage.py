"""Age-specific substance use rates."""

import pandas as pd

from ._containers import DescriptiveResult


def substance_by_age(
    df: pd.DataFrame,
    use_col: str = "substance_use",
    age_col: str = "age_group",
) -> DescriptiveResult:
    """Compute age-specific substance use rates.

    Parameters
    ----------
    df : DataFrame
        Must contain *use_col* (binary 0/1) and *age_col*.
    use_col : str
        Column with binary substance use indicator.
    age_col : str
        Column with age group labels.

    Returns
    -------
    DescriptiveResult
    """
    if use_col not in df.columns:
        raise ValueError(f"Column '{use_col}' not found")
    if age_col not in df.columns:
        raise ValueError(f"Column '{age_col}' not found")

    grouped = df.groupby(age_col)[use_col]
    rates = grouped.mean()
    counts = grouped.sum()
    totals = grouped.count()

    tbl = pd.DataFrame({"rate": rates, "n_users": counts, "n_total": totals})

    return DescriptiveResult(
        name="substance_by_age",
        value=tbl,
        extra={"n_groups": len(tbl), "overall_rate": float(df[use_col].mean())},
    )


suage = substance_by_age


def cheatsheet() -> str:
    return "substance_by_age({}) -> Age-specific substance use rates."
