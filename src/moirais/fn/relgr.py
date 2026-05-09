# moirais.fn — function file (hadesllm/moirais)
"""Reliability growth (Duane/AMSAA model). 'That which does not kill us makes us stronger. — Friedrich Nietzsche'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def reliability_growth(
    failures: np.ndarray,
    intervals: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Duane/AMSAA reliability growth model.

    Fits the Crow-AMSAA (NHPP power-law) model:

    .. math::

        N(t) = \lambda t^{\beta}

    where :math:`\lambda` is the intensity scale and :math:`\beta`
    is the growth parameter. :math:`\beta < 1` indicates reliability
    improvement; :math:`\beta > 1` indicates degradation.

    Parameters
    ----------
    failures : ndarray, shape (n_intervals,)
        Cumulative failure counts at each time interval.
    intervals : ndarray or None, shape (n_intervals,)
        Cumulative time at end of each interval. If None, uses
        1, 2, 3, ... as interval endpoints.

    Returns
    -------
    DescriptiveResult
        name='Reliability Growth', value=beta (growth rate),
        extra has 'beta', 'lambda_', 'instantaneous_mtbf',
        'cumulative_mtbf', 'growing' (bool).

    References
    ----------
    Crow, L.H. (1975). Reliability analysis for complex, repairable
    systems. In *Reliability and Biometry*, SIAM, 379-410.

    Duane, J.T. (1964). Learning curve approach to reliability
    monitoring. *IEEE Transactions on Aerospace*, 2(2), 563-566.
    doi:10.1109/TA.1964.4319640
    """
    N = np.asarray(failures, dtype=np.float64).ravel()
    n = len(N)

    if intervals is None:
        T = np.arange(1, n + 1, dtype=np.float64)
    else:
        T = np.asarray(intervals, dtype=np.float64).ravel()

    if len(T) != n:
        raise ValueError("failures and intervals must have same length.")

    mask = (N > 0) & (T > 0)
    if mask.sum() < 2:
        return DescriptiveResult(
            name="Reliability Growth",
            value=float("nan"),
            extra={"beta": float("nan"), "lambda_": float("nan"), "growing": False, "n_intervals": n},
        )

    log_T = np.log(T[mask])
    log_N = np.log(N[mask])

    m = mask.sum()
    beta = (m * np.sum(log_T * log_N) - np.sum(log_T) * np.sum(log_N)) / (m * np.sum(log_T**2) - np.sum(log_T) ** 2)
    log_lam = (np.sum(log_N) - beta * np.sum(log_T)) / m
    lam = np.exp(log_lam)

    T_last = T[-1]
    N_last = N[-1]
    cum_mtbf = T_last / N_last if N_last > 0 else float("inf")
    inst_mtbf = (1.0 / (lam * beta)) * T_last ** (1 - beta) if (lam * beta) > 0 else float("inf")

    return DescriptiveResult(
        name="Reliability Growth",
        value=float(beta),
        extra={
            "beta": float(beta),
            "lambda_": float(lam),
            "instantaneous_mtbf": float(inst_mtbf),
            "cumulative_mtbf": float(cum_mtbf),
            "growing": bool(beta < 1),
            "n_intervals": n,
        },
    )


relgr = reliability_growth


def cheatsheet() -> str:
    return "reliability_growth({}) -> Reliability growth (Duane/AMSAA model). 'The dark side cloud"
