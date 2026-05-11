# morie.fn — function file (hadesllm/morie)
"""Nelson-Aalen cumulative hazard estimator."""

import numpy as np

from morie.fn._containers import SurvivalResult


def nels(
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 0.05,
) -> SurvivalResult:
    """
    Nelson-Aalen estimator of the cumulative hazard function.

    .. math::

        \\hat{H}(t) = \\sum_{t_i \\le t} \\frac{d_i}{n_i}

    where :math:`d_i` is the number of events at :math:`t_i` and :math:`n_i`
    is the number at risk just before :math:`t_i`.

    :param time: 1-D array of observed times.
    :param event: 1-D binary array (1 = event, 0 = censored).
    :param alpha: Significance level for confidence interval. Default 0.05.
    :return: :class:`SurvivalResult` with ``times``, ``cumulative_hazard``
        (stored in *survival* field), ``ci_lower``, ``ci_upper``.
    :raises ValueError: If arrays differ in length or are empty.

    References
    ----------
    Nelson, W. (1972). Theory and estimation of hazard rate functions.
    *Biometrika*, 59(2), 313-326.

    Aalen, O. (1978). Nonparametric inference for a family of counting
    processes. *Annals of Statistics*, 6(4), 701-726.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if len(time) != len(event):
        raise ValueError(f"time and event must have same length, got {len(time)} and {len(event)}.")
    if len(time) == 0:
        raise ValueError("Input arrays must not be empty.")

    n = len(time)
    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]

    unique_times = np.unique(t_sorted)
    cum_hazard = np.zeros(len(unique_times))
    ci_lo = np.zeros(len(unique_times))
    ci_hi = np.zeros(len(unique_times))
    z = __import__("scipy").stats.norm.ppf(1 - alpha / 2)

    H = 0.0
    var_H = 0.0
    n_events_total = 0
    n_censored_total = 0

    for i, t_i in enumerate(unique_times):
        at_risk = np.sum(t_sorted >= t_i)
        d_i = np.sum((t_sorted == t_i) & (e_sorted == 1))
        c_i = np.sum((t_sorted == t_i) & (e_sorted == 0))
        n_events_total += int(d_i)
        n_censored_total += int(c_i)
        if at_risk > 0:
            H += d_i / at_risk
            var_H += d_i / (at_risk**2)
        cum_hazard[i] = H
        se_H = np.sqrt(var_H)
        ci_lo[i] = max(H - z * se_H, 0.0)
        ci_hi[i] = H + z * se_H

    return SurvivalResult(
        name="Nelson-Aalen",
        times=unique_times,
        survival=cum_hazard,  # cumulative hazard stored here
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_events=n_events_total,
        n_censored=n_censored_total,
        extra={"type": "cumulative_hazard"},
    )


def cheatsheet() -> str:
    return "nels({}) -> Nelson-Aalen cumulative hazard estimator."
