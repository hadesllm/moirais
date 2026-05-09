# moirais.fn — function file (hadesllm/moirais)
"""Competing risks — Cumulative Incidence Function (CIF)."""

import numpy as np


def crisk(
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Cumulative Incidence Function for competing risks.

    Uses the Aalen-Johansen estimator.  For cause *k*:

    .. math::

        \\hat{F}_k(t) = \\sum_{t_i \\le t}
            \\hat{S}(t_{i-1})\\,\\frac{d_{ik}}{n_i}

    where :math:`\\hat{S}` is the overall KM survival (treating all event
    types as events) and :math:`d_{ik}` is the count of cause-*k* events
    at :math:`t_i`.

    :param time: 1-D array of observed times.
    :param event: 1-D integer array: 0 = censored, 1 = event type 1,
        2 = event type 2 (supports arbitrary positive integers for types).
    :param alpha: Significance level for Greenwood-type SEs. Default 0.05.
    :return: dict with ``times``, ``cif`` (dict mapping event type to
        cumulative incidence array), ``se`` (dict), ``n``, ``n_events``
        (dict by type).
    :raises ValueError: On empty inputs or mismatched lengths.

    References
    ----------
    Aalen, O. O. & Johansen, S. (1978). An empirical transition matrix for
    non-homogeneous Markov chains. *Scandinavian Journal of Statistics*,
    5(3), 141-150.

    Fine, J. P. & Gray, R. J. (1999). A proportional hazards model for
    the sub-distribution of a competing risk. *JASA*, 94(446), 496-509.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if len(time) != len(event):
        raise ValueError(f"time and event must have same length, got {len(time)} and {len(event)}.")
    if len(time) == 0:
        raise ValueError("Input arrays must not be empty.")

    n = len(time)
    event_types = sorted(set(event[event > 0]))
    if len(event_types) == 0:
        raise ValueError("No events observed (all censored).")

    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]

    unique_times = np.unique(t_sorted)
    n_times = len(unique_times)
    z = __import__("scipy").stats.norm.ppf(1 - alpha / 2)

    # Overall KM survival (all event types as events)
    S = np.ones(n_times + 1)  # S[0] = 1 (before first time)
    at_risk_arr = np.zeros(n_times)
    for i, t_i in enumerate(unique_times):
        n_i = np.sum(t_sorted >= t_i)
        d_i = np.sum((t_sorted == t_i) & (e_sorted > 0))
        at_risk_arr[i] = n_i
        if n_i > 0:
            S[i + 1] = S[i] * (1 - d_i / n_i)
        else:
            S[i + 1] = S[i]

    # CIF for each event type
    cif_dict = {}
    se_dict = {}
    n_events_dict = {}
    for k in event_types:
        cif_k = np.zeros(n_times)
        var_k = np.zeros(n_times)
        F_k = 0.0
        v_k = 0.0
        for i, t_i in enumerate(unique_times):
            n_i = at_risk_arr[i]
            d_ik = np.sum((t_sorted == t_i) & (e_sorted == k))
            if n_i > 0:
                h_ik = d_ik / n_i
                F_k += S[i] * h_ik
                # Approximate variance
                v_k += (S[i] ** 2) * h_ik * (1 - h_ik) / max(n_i, 1)
            cif_k[i] = F_k
            var_k[i] = v_k
        cif_dict[int(k)] = cif_k
        se_dict[int(k)] = np.sqrt(var_k)
        n_events_dict[int(k)] = int(np.sum(event == k))

    return {
        "times": unique_times,
        "cif": cif_dict,
        "se": se_dict,
        "n": int(n),
        "n_events": n_events_dict,
    }


def cheatsheet() -> str:
    return "crisk({}) -> Competing risks — Cumulative Incidence Function (CIF)."
