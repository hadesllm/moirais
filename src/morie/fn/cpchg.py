# morie.fn -- function file (hadesllm/morie)
"""Change point detection via PELT algorithm."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["cpchg", "changepoint_pelt"]


def changepoint_pelt(
    y,
    penalty: str | float = "bic",
    min_size: int = 2,
) -> DescriptiveResult:
    """Detect change points using PELT (Pruned Exact Linear Time).

    Minimises the total segmentation cost plus a penalty for the number of
    change points.  Cost is the within-segment negative log-likelihood under
    a Gaussian mean-shift model (equivalent to sum of squared deviations from
    the segment mean).

    Dynamic-programming recursion::

        F[t] = min_{s < t} { F[s] + cost(s, t) + beta }

    where ``beta`` is the per-change-point penalty.

    Parameters
    ----------
    y : array-like
        Univariate time series (n,).
    penalty : str or float
        Penalty per change point.
        ``'bic'`` uses ``log(n)`` (Schwarz criterion).
        ``'aic'`` uses ``2.0``.
        A numeric value uses that constant directly.
        Default ``'bic'``.
    min_size : int
        Minimum number of observations per segment.  Default 2.

    Returns
    -------
    DescriptiveResult
        value: float -- number of detected change points.
        extra keys:
          'changepoints' : 1-D int array of change point indices (positions
                           in the *original* series where a new segment begins,
                           excluding 0 and n).
          'n_changepoints': int.
          'total_cost'   : float -- optimal total cost including penalties.
          'segments'     : list of (start, end) tuples (half-open, start inclusive).
          'penalty_value': float -- numeric penalty used.

    Raises
    ------
    ValueError
        If series is too short for the requested min_size.

    References
    ----------
    Killick R., Fearnhead P. & Eckley I.A. (2012). Optimal detection of
    changepoints with a linear computational cost.
    Journal of the American Statistical Association, 107(500), 1590-1598.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}.")
    if n < 2 * min_size:
        raise ValueError(
            f"Need n >= 2*min_size = {2*min_size}; got n={n}."
        )

    if isinstance(penalty, str):
        if penalty == "bic":
            beta = float(np.log(n))
        elif penalty == "aic":
            beta = 2.0
        else:
            raise ValueError(f"Unknown string penalty '{penalty}'; use 'bic', 'aic', or a float.")
    else:
        beta = float(penalty)

    # Pre-compute prefix sums for O(1) segment cost computation.
    # cost(s, t) = sum_{i=s}^{t-1} (y[i] - mean)^2
    #            = sum_sq[s:t] - (sum[s:t])^2 / (t - s)
    # for segment [s, t) (t not included).
    cumsum = np.zeros(n + 1)
    cumsum2 = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(y)
    cumsum2[1:] = np.cumsum(y ** 2)

    def _cost(s: int, t: int) -> float:
        """Within-segment SSQ cost for observations y[s:t]."""
        length = t - s
        if length <= 0:
            return 0.0
        s_sum = cumsum[t] - cumsum[s]
        s_sq = cumsum2[t] - cumsum2[s]
        return float(s_sq - s_sum ** 2 / length)

    # PELT dynamic programming (without full pruning for simplicity;
    # still O(n^2) worst case but correct and exact).
    INF = float("inf")
    F = np.full(n + 1, INF)
    F[0] = -beta  # so F[0] + beta + cost(0, t) = cost(0, t) for first seg
    last_cp = np.zeros(n + 1, dtype=int)

    for t in range(min_size, n + 1):
        best = INF
        best_s = 0
        for s in range(0, t - min_size + 1):
            seg_len = t - s
            if seg_len < min_size:
                continue
            candidate = F[s] + beta + _cost(s, t)
            if candidate < best:
                best = candidate
                best_s = s
        if best < INF:
            F[t] = best
            last_cp[t] = best_s

    # Back-track to find change points.
    cps = []
    t = n
    while t > 0:
        s = last_cp[t]
        if s > 0:
            cps.append(s)
        t = s

    cps_arr = np.array(sorted(cps), dtype=int)

    # Build segment list.
    breakpoints = np.concatenate([[0], cps_arr, [n]])
    segments = [
        (int(breakpoints[i]), int(breakpoints[i + 1]))
        for i in range(len(breakpoints) - 1)
    ]

    return DescriptiveResult(
        name="changepoint_pelt",
        value=float(len(cps_arr)),
        extra={
            "changepoints": cps_arr,
            "n_changepoints": int(len(cps_arr)),
            "total_cost": float(F[n]),
            "segments": segments,
            "penalty_value": beta,
        },
    )


cpchg = changepoint_pelt


def cheatsheet() -> str:
    return "changepoint_pelt(y, penalty='bic', min_size=2) -> PELT change point detection."
