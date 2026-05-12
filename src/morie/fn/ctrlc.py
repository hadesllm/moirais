# morie.fn — function file (hadesllm/morie)
"""Comparisons with a control (Gibbons Ch 10.7).

Nonparametric many-to-one comparison: each treatment group i is
compared with the control via Mann-Whitney U; p-values are
adjusted for multiplicity using Bonferroni (default).  This is
the nonparametric analogue of Dunnett's test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["control_comparison"]


def control_comparison(groups, control_index: int = 0, adjust: str = "bonferroni"):
    """Many-to-one nonparametric comparison to a control group.

    Parameters
    ----------
    groups : sequence of array-like
        Sequence whose first element (or ``control_index``-th) is the
        control sample; remaining are treatment samples.
    control_index : int
        Index of the control group within ``groups``.  Default 0.
    adjust : {"bonferroni", "none"}
        Multiplicity adjustment.

    Returns
    -------
    RichResult with payload:
        statistic    : array of Mann-Whitney U statistics (vs. control)
        p_value      : array of *unadjusted* p-values
        p_adjusted   : array of multiplicity-adjusted p-values
        n            : array of treatment-group sizes
        k            : number of comparisons
        control_n    : control sample size
        method       : descriptor
    """
    if not isinstance(groups, (list, tuple)) or len(groups) < 2:
        return RichResult(payload={
            "statistic": np.array([]), "p_value": np.array([]),
            "p_adjusted": np.array([]),
            "n": np.array([]), "k": 0, "control_n": 0,
            "method": "Many-to-one control comparison",
        })
    arrs = [np.asarray(g, dtype=float).ravel() for g in groups]
    ctrl = arrs[control_index]
    trts = [a for i, a in enumerate(arrs) if i != control_index]
    k = len(trts)
    Us, ps = [], []
    for t in trts:
        if min(len(ctrl), len(t)) < 2:
            Us.append(np.nan)
            ps.append(np.nan)
            continue
        r = stats.mannwhitneyu(ctrl, t, alternative="two-sided")
        Us.append(float(r.statistic))
        ps.append(float(r.pvalue))
    Us = np.array(Us)
    ps = np.array(ps)
    if adjust == "bonferroni":
        p_adj = np.minimum(ps * k, 1.0)
    elif adjust == "none":
        p_adj = ps.copy()
    else:
        raise ValueError("adjust must be 'bonferroni' or 'none'")
    return RichResult(payload={
        "statistic": Us,
        "p_value": ps,
        "p_adjusted": p_adj,
        "n": np.array([len(t) for t in trts]),
        "k": k,
        "control_n": len(ctrl),
        "adjust": adjust,
        "method": "Nonparametric many-to-one (Mann-Whitney) vs. control",
    })


def cheatsheet():
    return "ctrlc: Nonparametric many-to-one comparison to control"


# CANONICAL TEST
# >>> control_comparison([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
# Both treatments lie strictly above control -> small Bonferroni p-values
