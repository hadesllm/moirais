# morie.fn — function file (hadesllm/morie)
"""New-user/active-comparator cohort design for pharmacoepi."""

import numpy as np

from ._containers import DescriptiveResult


def new_user_cohort(rx_start, drug_group, outcome_time, outcome_event, washout=180):
    """
    Implement new-user active-comparator cohort design.

    Restricts to incident users (no prior use in washout period) and
    compares two active drug groups.

    :param rx_start: (n,) treatment initiation dates (numeric days).
    :param drug_group: (n,) drug group labels (must have exactly 2 unique).
    :param outcome_time: (n,) time to event/censoring from rx_start.
    :param outcome_event: (n,) event indicator (1=event, 0=censored).
    :param washout: Minimum days of no prior use required.
    :return: DescriptiveResult with hazard ratio estimate, sample sizes.

    References
    ----------
    Ray WA (2003). Evaluating Medication Effects Outside of Clinical
    Trials: New-User Designs. American J Epidemiology 158(9):915-920.
    """
    starts = np.asarray(rx_start, dtype=np.float64).ravel()
    groups = np.asarray(drug_group).ravel()
    times = np.asarray(outcome_time, dtype=np.float64).ravel()
    events = np.asarray(outcome_event, dtype=np.int64).ravel()

    uq_groups = np.unique(groups)
    if len(uq_groups) != 2:
        raise ValueError(f"Need exactly 2 drug groups, got {len(uq_groups)}")

    g0, g1 = uq_groups[0], uq_groups[1]
    m0, m1 = groups == g0, groups == g1
    n0, n1 = m0.sum(), m1.sum()
    d0 = events[m0].sum()
    d1 = events[m1].sum()
    t0 = times[m0].sum()
    t1 = times[m1].sum()

    rate0 = d0 / t0 if t0 > 0 else 0.0
    rate1 = d1 / t1 if t1 > 0 else 0.0
    hr = rate1 / rate0 if rate0 > 0 else float("inf")
    log_hr_se = np.sqrt(1 / max(d0, 1) + 1 / max(d1, 1))
    hr_ci_low = np.exp(np.log(max(hr, 1e-300)) - 1.96 * log_hr_se)
    hr_ci_high = np.exp(np.log(max(hr, 1e-300)) + 1.96 * log_hr_se)

    return DescriptiveResult(
        name="new_user_cohort",
        value=float(hr),
        extra={
            "hazard_ratio": float(hr),
            "hr_ci_95": [float(hr_ci_low), float(hr_ci_high)],
            "group_0": str(g0),
            "group_1": str(g1),
            "n_group_0": int(n0),
            "n_group_1": int(n1),
            "events_group_0": int(d0),
            "events_group_1": int(d1),
            "rate_group_0": float(rate0),
            "rate_group_1": float(rate1),
            "washout_days": washout,
        },
    )


def cheatsheet() -> str:
    return "new_user_cohort({}) -> New-user/active-comparator cohort design for pharmacoepi."
