# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bai-Perron structural break detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["bkpts", "bai_perron"]


def bai_perron(
    y,
    x=None,
    max_breaks: int = 3,
    min_size: int | None = None,
    sig_level: float = 0.05,
) -> DescriptiveResult:
    """Detect structural breaks via the Bai-Perron sequential F-test.

    Estimates the number of structural breaks in a linear regression by the
    sequential ``sup-F(m|m-1)`` test procedure.  For each candidate number of
    breaks *m*, the globally optimal break dates are found by dynamic
    programming, and an F-statistic compares the (m-1)-break fit against the
    m-break fit.

    Model for segment j: ``y[t] = X[t] @ beta_j + epsilon[t]``

    Parameters
    ----------
    y : array-like
        Dependent variable (n,).
    x : array-like or None
        Regressors (n,) or (n, k).  None -> constant-only (k=1).
    max_breaks : int
        Maximum number of breaks to consider.  Default 3.
    min_size : int or None
        Minimum segment length.  If None uses ``max(k+1, ceil(0.15*n))``.
    sig_level : float
        Significance level for sequential sup-F test.  Default 0.05.

    Returns
    -------
    DescriptiveResult
        value: float -- estimated number of breaks.
        extra keys:
          'break_dates'  : int array of break indices (segment start positions
                           in y, excluding 0 and n).
          'n_breaks'     : int.
          'rss_by_breaks': dict {m: RSS}.
          'f_stats'      : list of sup-F statistics.
          'coef_by_seg'  : list of coefficient arrays per segment.
          'segments'     : list of (start, end) half-open tuples.

    References
    ----------
    Bai J. & Perron P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47-78.

    Bai J. & Perron P. (2003). Computation and analysis of multiple structural
    change models. Journal of Applied Econometrics, 18(1), 1-22.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if x is None:
        X = np.ones((n, 1))
    else:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    k = X.shape[1]
    if X.shape[0] != n:
        raise ValueError(f"x must have {n} rows, got {X.shape[0]}.")
    if max_breaks < 1:
        raise ValueError(f"max_breaks must be >= 1, got {max_breaks}.")
    if not (0.0 < sig_level < 1.0):
        raise ValueError(f"sig_level must be in (0,1), got {sig_level}.")

    if min_size is None:
        min_size = max(k + 1, int(np.ceil(0.15 * n)))
    min_size = max(min_size, k + 1)

    max_breaks = min(max_breaks, max(0, n // min_size - 1))

    def _seg_ols(s: int, e: int):
        """(coef, rss) for y[s:e] ~ X[s:e]."""
        Xs, ys = X[s:e], y[s:e]
        if len(ys) < k + 1:
            return np.zeros(k), float("inf")
        coef, res, rank, _ = np.linalg.lstsq(Xs, ys, rcond=None)
        rss = float(res[0]) if len(res) > 0 else float(np.sum((ys - Xs @ coef) ** 2))
        return coef, rss

    # Pre-compute segment RSS (lower triangular).
    seg_rss = {}
    for s in range(n - min_size + 1):
        for e in range(s + min_size, n + 1):
            seg_rss[(s, e)] = _seg_ols(s, e)[1]

    INF = float("inf")
    rss_by_m = {0: seg_rss.get((0, n), INF)}

    # dp_list[m][t] = (min_rss_for_m+1_segments_covering_y[0:t], break_dates_list)
    # For m=0: single segment [0, t) for each valid t.
    dp0 = {}
    for t in range(min_size, n + 1):
        v = seg_rss.get((0, t), INF)
        if v < INF:
            dp0[t] = (v, [])
    dp_list = [dp0]

    for m in range(1, max_breaks + 1):
        dp_curr = {}
        for e in range((m + 1) * min_size, n + 1):
            best = INF
            best_breaks = None
            # Last segment starts at s, covers y[s:e].
            for s in range(m * min_size, e - min_size + 1):
                if s not in dp_list[m - 1]:
                    continue
                prev_rss, prev_bps = dp_list[m - 1][s]
                seg_cost = seg_rss.get((s, e), INF)
                candidate = prev_rss + seg_cost
                if candidate < best:
                    best = candidate
                    best_breaks = prev_bps + [s]
            if best_breaks is not None:
                dp_curr[e] = (best, best_breaks)
        dp_list.append(dp_curr)
        rss_by_m[m] = dp_curr[n][0] if n in dp_curr else INF

    # Sequential sup-F test (Bai & Perron 2003, Table I, eps=0.15).
    _cv = {0.10: [7.04, 8.51, 9.41], 0.05: [8.58, 10.13, 11.14], 0.01: [12.29, 14.20, 15.09]}
    if sig_level <= 0.01:
        cv_row = _cv[0.01]
    elif sig_level <= 0.05:
        cv_row = _cv[0.05]
    else:
        cv_row = _cv[0.10]

    estimated_breaks = 0
    f_stats = []
    for m in range(1, max_breaks + 1):
        rss_m = rss_by_m.get(m, INF)
        rss_m1 = rss_by_m.get(m - 1, INF)
        if rss_m >= rss_m1 or rss_m1 == INF:
            f_stats.append(0.0)
            break
        n_use = max(n - (m + 1) * k, 1)
        # Guard against perfect fit (rss_m == 0) and degenerate k == 0.
        # A perfect fit means the improvement-ratio is effectively infinite,
        # which is "accept this many breaks"; use a large sentinel.
        if rss_m <= 0 or k <= 0:
            f_stat = 1e12
        else:
            f_stat = float(((rss_m1 - rss_m) / k) / (rss_m / n_use))
        f_stats.append(f_stat)
        cv_val = cv_row[m - 1] if m - 1 < len(cv_row) else cv_row[-1]
        if f_stat > cv_val:
            estimated_breaks = m
        else:
            break

    if estimated_breaks > 0 and n in dp_list[estimated_breaks]:
        break_dates = np.array(dp_list[estimated_breaks][n][1], dtype=int)
    else:
        break_dates = np.array([], dtype=int)

    all_bps = np.concatenate([[0], break_dates, [n]])
    segments = [(int(all_bps[i]), int(all_bps[i + 1])) for i in range(len(all_bps) - 1)]
    coef_by_seg = [_seg_ols(s, e)[0].tolist() for s, e in segments]

    return DescriptiveResult(
        name="bai_perron",
        value=float(estimated_breaks),
        extra={
            "break_dates": break_dates,
            "n_breaks": int(estimated_breaks),
            "rss_by_breaks": rss_by_m,
            "f_stats": f_stats,
            "coef_by_seg": coef_by_seg,
            "segments": segments,
        },
    )


bkpts = bai_perron
breakpoint_detection = bai_perron


def cheatsheet() -> str:
    return "bai_perron(y, max_breaks=3) -> Bai-Perron structural break detection."
