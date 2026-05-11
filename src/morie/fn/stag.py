"""Staggered Difference-in-Differences (Callaway-Sant'Anna style)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def staggered_did(
    outcome: Union[list, np.ndarray],
    group: Union[list, np.ndarray],
    time: Union[list, np.ndarray],
    first_treat: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Staggered DiD estimator following Callaway & Sant'Anna (2021).

    Computes group-time average treatment effects ATT(g, t) for each
    cohort g (units first treated at time g) and each post-treatment
    period t >= g, using a never-treated or not-yet-treated comparison
    group.

    The overall ATT is the (sample-size) weighted average of ATT(g,t).

    Simplified implementation: uses simple DiD within each (g, t) cell
    with never-treated units as the comparison group (first_treat == 0 or
    first_treat > max(time)).

    :param outcome: Outcome variable.
    :param group: Unit identifier (used to identify panel structure).
    :param time: Time period.
    :param first_treat: Period of first treatment for each unit (0 or
        large value = never treated).
    :return: Dictionary with att_gt (list of dicts), att_overall, n_groups.
    :raises ValueError: If no never-treated comparison group found.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*, 225(2),
    200--230.
    """
    y = np.asarray(outcome, dtype=float)
    g_id = np.asarray(group)
    t_arr = np.asarray(time, dtype=int)
    ft = np.asarray(first_treat, dtype=int)
    n = len(y)
    if len(g_id) != n or len(t_arr) != n or len(ft) != n:
        raise ValueError("All arrays must have the same length.")

    max_t = int(np.max(t_arr))
    # Never-treated: first_treat == 0 or > max_t
    never_treated_mask = (ft == 0) | (ft > max_t)
    if not np.any(never_treated_mask):
        raise ValueError("No never-treated comparison group found (first_treat == 0 or > max time).")

    # Unique treatment cohorts (excluding never-treated)
    cohorts = sorted(set(ft[~never_treated_mask]))
    periods = sorted(set(t_arr))

    att_gt_list = []

    for cohort in cohorts:
        # Pre-period: last period before treatment
        pre_periods = [p for p in periods if p < cohort]
        if not pre_periods:
            continue
        pre_t = max(pre_periods)

        for post_t in [p for p in periods if p >= cohort]:
            # Treated units in this cohort
            treat_mask = ft == cohort
            # Outcomes for treated in post and pre
            y_treat_post = y[treat_mask & (t_arr == post_t)]
            y_treat_pre = y[treat_mask & (t_arr == pre_t)]
            # Never-treated comparison
            y_ctrl_post = y[never_treated_mask & (t_arr == post_t)]
            y_ctrl_pre = y[never_treated_mask & (t_arr == pre_t)]

            if len(y_treat_post) == 0 or len(y_treat_pre) == 0:
                continue
            if len(y_ctrl_post) == 0 or len(y_ctrl_pre) == 0:
                continue

            att = (np.mean(y_treat_post) - np.mean(y_treat_pre)) - (np.mean(y_ctrl_post) - np.mean(y_ctrl_pre))
            att_gt_list.append(
                {
                    "cohort": int(cohort),
                    "time": int(post_t),
                    "att": float(att),
                    "n_treated": len(y_treat_post),
                }
            )

    # Overall ATT: weighted average by n_treated
    if att_gt_list:
        total_n = sum(d["n_treated"] for d in att_gt_list)
        att_overall = sum(d["att"] * d["n_treated"] for d in att_gt_list) / total_n
    else:
        att_overall = float("nan")

    return {
        "att_gt": att_gt_list,
        "att_overall": float(att_overall),
        "n_groups": len(cohorts),
    }


stag = staggered_did


def cheatsheet() -> str:
    return "staggered_did({}) -> Staggered Difference-in-Differences (Callaway-Sant'Anna styl"
