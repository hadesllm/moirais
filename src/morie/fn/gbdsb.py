# morie.fn — function file (hadesllm/morie)
"""Burden by subgroup (GBD-style)."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def gbd_subgroup(
    dalys: list | np.ndarray,
    groups: list | np.ndarray,
) -> DescriptiveResult:
    """Compute burden by subgroup (sex, region, SES).

    Parameters
    ----------
    dalys : array-like
        DALY values per individual/unit.
    groups : array-like
        Group labels.

    Returns
    -------
    DescriptiveResult
    """
    d = np.asarray(dalys, dtype=float)
    g = np.asarray(groups, dtype=str)
    if len(d) != len(g):
        raise ValueError("dalys and groups must match")

    df = pd.DataFrame({"daly": d, "group": g})
    summary = df.groupby("group")["daly"].agg(["sum", "mean", "count"])
    total = float(d.sum())
    summary["pct"] = summary["sum"] / total * 100

    return DescriptiveResult(
        name="gbd_subgroup",
        value=summary,
        extra={"total_dalys": total, "n_groups": len(summary)},
    )


gbdsb = gbd_subgroup


def cheatsheet() -> str:
    return "gbd_subgroup({}) -> Burden by subgroup (GBD-style)."
