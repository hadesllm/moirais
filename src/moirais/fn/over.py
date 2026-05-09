# moirais.fn — function file (hadesllm/moirais)
"""Overlap / common support assessment for propensity scores."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def overlap_diagnostics(
    ps_treated: Union[list, np.ndarray],
    ps_control: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Assess propensity score overlap (common support) between groups.

    Computes the overlap region, percentage of observations that would be
    trimmed by standard rules, and a two-sample Kolmogorov-Smirnov test
    for distributional differences.

    :param ps_treated: Propensity scores for treated group.
    :param ps_control: Propensity scores for control group.
    :return: Dictionary with overlap_range (tuple), pct_trimmed (float),
        ks_stat (float), ks_p (float), n_treated, n_control.
    :raises ValueError: If either array is empty.

    References
    ----------
    Imbens, G. W. (2015). Matching methods in practice: Three examples.
    *Journal of Human Resources*, 50(2), 373--419.
    """
    from scipy import stats as _st

    pt = np.asarray(ps_treated, dtype=float)
    pc = np.asarray(ps_control, dtype=float)
    if len(pt) == 0 or len(pc) == 0:
        raise ValueError("Both arrays must be non-empty.")

    # Overlap region: intersection of ranges
    lo = max(float(np.min(pt)), float(np.min(pc)))
    hi = min(float(np.max(pt)), float(np.max(pc)))
    overlap_range = (lo, hi) if lo < hi else (0.0, 0.0)

    # Pct trimmed if we keep only the overlap region
    n_total = len(pt) + len(pc)
    in_overlap_t = np.sum((pt >= lo) & (pt <= hi))
    in_overlap_c = np.sum((pc >= lo) & (pc <= hi))
    n_trimmed = n_total - int(in_overlap_t) - int(in_overlap_c)
    pct_trimmed = 100.0 * n_trimmed / n_total if n_total > 0 else 0.0

    # KS test
    ks_stat, ks_p = _st.ks_2samp(pt, pc)

    return {
        "overlap_range": overlap_range,
        "pct_trimmed": float(pct_trimmed),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "n_treated": len(pt),
        "n_control": len(pc),
    }


over = overlap_diagnostics


def cheatsheet() -> str:
    return "overlap_diagnostics({}) -> Overlap / common support assessment for propensity scores."
