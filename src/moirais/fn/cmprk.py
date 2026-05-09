# moirais.fn — function file (hadesllm/moirais)
"""Competing risks — cumulative incidence via Aalen-Johansen."""

import numpy as np

from ._containers import DescriptiveResult


def competing_risks(time, event, cause_of_interest=1):
    """
    Cumulative incidence function for competing risks (Aalen-Johansen).

    :param time: (n,) event/censoring times.
    :param event: (n,) event indicator (0=censored, 1,2,...=cause).
    :param cause_of_interest: Which cause to compute CIF for.
    :return: DescriptiveResult with CIF, times, overall survival.

    References
    ----------
    Aalen OO, Johansen S (1978). An Empirical Transition Matrix for
    Non-Homogeneous Markov Chains. Scandinavian J Statistics 5(3):141-150.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.int64).ravel()
    order = np.argsort(time)
    time, event = time[order], event[order]
    unique_times = np.unique(time[event > 0])
    n = len(time)

    cif = np.zeros(len(unique_times))
    surv = 1.0
    at_risk = n

    for i, t in enumerate(unique_times):
        mask = time == t
        d_cause = np.sum((event == cause_of_interest) & mask)
        d_all = np.sum((event > 0) & mask)
        if at_risk > 0:
            cif[i] = cif[i - 1] + surv * d_cause / at_risk if i > 0 else surv * d_cause / at_risk
            surv *= 1 - d_all / at_risk
        else:
            cif[i] = cif[i - 1] if i > 0 else 0.0
        at_risk -= np.sum(mask)

    return DescriptiveResult(
        name="competing_risks",
        value=float(cif[-1]) if len(cif) > 0 else 0.0,
        extra={
            "times": unique_times.tolist(),
            "cif": cif.tolist(),
            "final_cif": float(cif[-1]) if len(cif) > 0 else 0.0,
            "cause": cause_of_interest,
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "competing_risks({}) -> Competing risks — cumulative incidence via Aalen-Johansen."
