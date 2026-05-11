# morie.fn — function file (hadesllm/morie)
"""Restricted mean survival time."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rmst_estimate(time, event, *, tau: float | None = None) -> DescriptiveResult:
    """RMST: area under Kaplan-Meier curve up to time tau.

    Parameters
    ----------
    time : array-like
        Observed survival times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    tau : float, optional
        Restriction time. Default: max observed time.

    Returns
    -------
    DescriptiveResult
    """
    from morie.fn.km import kaplan_meier

    km_result = kaplan_meier(time, event)
    t = km_result.times
    s = km_result.survival
    if tau is None:
        tau = float(t[-1])
    mask = t <= tau
    t_trunc = np.append(t[mask], tau)
    s_trunc = np.append(s[mask], s[mask][-1] if mask.any() else 1.0)
    area = float(np.trapezoid(s_trunc, t_trunc))
    return DescriptiveResult(
        name="RMST",
        value=area,
        extra={"tau": tau, "n_events": km_result.n_events},
    )


rmst = rmst_estimate


def cheatsheet() -> str:
    return "rmst_estimate({}) -> Restricted mean survival time."
