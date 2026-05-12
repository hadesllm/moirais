"""Transmission tree reconstruction (Wallinga-Teunis method)."""

from __future__ import annotations

from typing import Any

import numpy as np


def transmission_tree(
    onset_times: np.ndarray,
    serial_interval_pmf: np.ndarray,
    *,
    ids: list | None = None,
) -> dict[str, Any]:
    r"""Reconstruct a transmission tree using the Wallinga-Teunis method.

    For each pair of cases (i, j), computes the relative likelihood
    that case j infected case i based on the serial interval
    distribution. The most likely infector for each case is identified.

    .. math::

        p_{ij} = \\frac{w(t_i - t_j)}{\\sum_{k \\neq i} w(t_i - t_k)}

    Parameters
    ----------
    onset_times : array_like
        Symptom onset times for each case (numeric, e.g. day number).
    serial_interval_pmf : array_like
        PMF of serial interval (index 0 = lag 0, index 1 = lag 1, ...).
    ids : list or None
        Case identifiers. If None, uses integer indices.

    Returns
    -------
    dict
        Keys: 'infector' (most likely infector index for each case),
              'probability_matrix' (n x n relative likelihood matrix),
              'Ri' (individual reproduction number per case),
              'n_cases'.

    References
    ----------
    Wallinga, J. & Teunis, P. (2004). Different epidemic curves for
    severe acute respiratory syndrome reveal similar impacts of control
    measures. American Journal of Epidemiology, 160(6), 509-516.
    """
    times = np.asarray(onset_times, dtype=float)
    w = np.asarray(serial_interval_pmf, dtype=float)

    n = len(times)
    if n < 2:
        raise ValueError("Need at least 2 cases.")

    if ids is None:
        ids = list(range(n))

    max_lag = len(w)
    P = np.zeros((n, n))

    for i in range(n):
        denom = 0.0
        for k in range(n):
            if k == i:
                continue
            lag = int(round(times[i] - times[k]))
            if 0 <= lag < max_lag:
                denom += w[lag]

        if denom > 0:
            for j in range(n):
                if j == i:
                    continue
                lag = int(round(times[i] - times[j]))
                if 0 <= lag < max_lag:
                    P[i, j] = w[lag] / denom

    infector = np.full(n, -1, dtype=int)
    for i in range(n):
        row = P[i].copy()
        row[i] = 0
        if np.any(row > 0):
            infector[i] = int(np.argmax(row))

    Ri = np.sum(P, axis=0)

    return {
        "infector": infector,
        "probability_matrix": P,
        "Ri": Ri,
        "n_cases": n,
        "ids": ids,
    }


tranm = transmission_tree


def cheatsheet() -> str:
    return "transmission_tree({}) -> Wallinga-Teunis transmission tree."
