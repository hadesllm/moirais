# morie.fn -- function file (hadesllm/morie)
"""Program effect by subgroup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def program_subgroup(
    df: pd.DataFrame,
    *,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    subgroup_col: str = "subgroup",
) -> DescriptiveResult:
    """Compute mean treatment effect within each subgroup.

    Parameters
    ----------
    df : DataFrame
    outcome_col, treatment_col, subgroup_col : str

    Returns
    -------
    DescriptiveResult
    """
    for c in [outcome_col, treatment_col, subgroup_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")

    effects = {}
    for grp, sub in df.groupby(subgroup_col):
        treated = sub[sub[treatment_col] == 1][outcome_col].values
        control = sub[sub[treatment_col] == 0][outcome_col].values
        if len(treated) > 0 and len(control) > 0:
            effects[str(grp)] = float(np.mean(treated) - np.mean(control))

    return DescriptiveResult(
        name="program_subgroup_effects",
        value=float(np.mean(list(effects.values()))) if effects else 0.0,
        extra={"subgroup_effects": effects, "n_subgroups": len(effects)},
    )


prgsb = program_subgroup


def cheatsheet() -> str:
    return "program_subgroup({}) -> Program effect by subgroup."
