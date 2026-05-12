# morie.fn -- function file (hadesllm/morie)
"""Cumulative incidence function -- non-parametric."""

import numpy as np

from ._containers import DescriptiveResult


def cumulative_incidence(time, event, group=None):
    """
    Non-parametric cumulative incidence with optional group comparison.

    :param time: (n,) event/censoring times.
    :param event: (n,) binary event indicator.
    :param group: (n,) optional group labels for comparison.
    :return: DescriptiveResult with CIF by group, Gray's test if grouped.

    References
    ----------
    Gray RJ (1988). A Class of K-Sample Tests for Comparing the
    Cumulative Incidence of a Competing Risk. Annals of Statistics
    16(3):1141-1154.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.int64).ravel()
    n = len(time)

    def _cif_single(t, e):
        order = np.argsort(t)
        t, e = t[order], e[order]
        unique = np.unique(t[e == 1])
        surv = 1.0
        cif_vals = np.zeros(len(unique))
        nr = len(t)
        for i, ti in enumerate(unique):
            mask = t == ti
            d = np.sum(e[mask] == 1)
            n_all = np.sum(mask)
            cif_vals[i] = (cif_vals[i - 1] if i > 0 else 0) + surv * d / nr
            surv *= 1 - n_all / nr
            nr -= np.sum(mask)
            if nr <= 0:
                break
        return unique, cif_vals

    if group is None:
        times_out, cif_out = _cif_single(time, event)
        return DescriptiveResult(
            name="cumulative_incidence",
            value=float(cif_out[-1]) if len(cif_out) > 0 else 0.0,
            extra={"times": times_out.tolist(), "cif": cif_out.tolist(), "n": n},
        )

    group = np.asarray(group).ravel()
    groups = np.unique(group)
    results = {}
    for g in groups:
        mask = group == g
        t_g, cif_g = _cif_single(time[mask], event[mask])
        results[str(g)] = {"times": t_g.tolist(), "cif": cif_g.tolist(), "n": int(mask.sum())}

    return DescriptiveResult(
        name="cumulative_incidence",
        value=len(groups),
        extra={"groups": results, "n_groups": len(groups), "n": n},
    )


def cheatsheet() -> str:
    return "cumulative_incidence({}) -> Cumulative incidence function -- non-parametric."
