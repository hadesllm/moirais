# morie.fn -- function file (hadesllm/morie)
"""Propensity score overlap check. 'Knowing yourself is the beginning of all wisdom. -- Aristotle'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ps_overlap(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
) -> DescriptiveResult:
    """
    Check overlap (positivity) of propensity score distributions.

    Reports the overlap region, c-statistic, and proportion of each
    group falling within the common support.

    :param ps_treated: Propensity scores for treated group.
    :param ps_control: Propensity scores for control group.
    :return: DescriptiveResult with overlap proportion as value.
    :raises ValueError: If arrays are empty.

    References
    ----------
    Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y., & van der Laan,
    M. J. (2012). Diagnosing and responding to violations in the positivity
    assumption. Statistical Methods in Medical Research, 21(1), 31--54.
    doi:10.1177/0962280210386207
    """
    t = np.asarray(ps_treated, dtype=float)
    c = np.asarray(ps_control, dtype=float)
    if t.size == 0 or c.size == 0:
        raise ValueError("Both ps_treated and ps_control must be non-empty.")

    common_min = max(np.min(t), np.min(c))
    common_max = min(np.max(t), np.max(c))

    if common_min >= common_max:
        overlap_t = 0.0
        overlap_c = 0.0
    else:
        overlap_t = float(np.mean((t >= common_min) & (t <= common_max)))
        overlap_c = float(np.mean((c >= common_min) & (c <= common_max)))

    all_ps = np.concatenate([t, c])
    all_labels = np.concatenate([np.ones(len(t)), np.zeros(len(c))])
    order = np.argsort(-all_ps)
    sorted_labels = all_labels[order]
    tpr_sum = np.cumsum(sorted_labels)
    n_pos = np.sum(sorted_labels)
    n_neg = len(sorted_labels) - n_pos
    c_stat = float(np.sum(tpr_sum[sorted_labels == 0]) / (n_pos * n_neg)) if n_pos > 0 and n_neg > 0 else 0.5

    return DescriptiveResult(
        name="PS Overlap",
        value=float(np.round(min(overlap_t, overlap_c), 4)),
        extra={
            "overlap_treated": float(np.round(overlap_t, 4)),
            "overlap_control": float(np.round(overlap_c, 4)),
            "common_support": (float(np.round(common_min, 4)), float(np.round(common_max, 4))),
            "c_statistic": float(np.round(c_stat, 4)),
            "n_treated": len(t),
            "n_control": len(c),
            "mean_ps_treated": float(np.round(np.mean(t), 4)),
            "mean_ps_control": float(np.round(np.mean(c), 4)),
        },
    )


psovl = ps_overlap


def cheatsheet() -> str:
    return "ps_overlap({}) -> Propensity score overlap check. 'Fear is the path to the dar"
