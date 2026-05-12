# morie.fn -- function file (hadesllm/morie)
"""Acquittal rate by offense type."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def court_acquittal(
    outcomes: list[str] | np.ndarray,
    offense_types: list[str] | np.ndarray,
) -> DescriptiveResult:
    """Compute acquittal rate overall and by offense type.

    Parameters
    ----------
    outcomes : array-like
        Case outcome labels (must include 'Acquitted' or similar).
    offense_types : array-like
        Corresponding offense type labels.

    Returns
    -------
    DescriptiveResult
    """
    if len(outcomes) != len(offense_types):
        raise ValueError("outcomes and offense_types must be same length")
    if len(outcomes) == 0:
        raise ValueError("Must have at least 1 case")
    import pandas as pd

    df = pd.DataFrame({"outcome": outcomes, "offense": offense_types})
    df["acquitted"] = df["outcome"].str.lower().str.contains("acquit", na=False).astype(int)
    overall = float(df["acquitted"].mean())
    by_offense = df.groupby("offense")["acquitted"].mean().to_dict()
    return DescriptiveResult(
        name="acquittal_rate",
        value=overall,
        extra={"overall_rate": overall, "by_offense": by_offense, "n": len(df)},
    )


crtaq = court_acquittal


def cheatsheet() -> str:
    return "court_acquittal({}) -> Acquittal rate by offense type."
