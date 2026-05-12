# morie.fn -- function file (hadesllm/morie)
"""Time series structural break detection + recovery. 'I am fire and life incarnate.' -- Phoenix"""

from __future__ import annotations

import numpy as np

from ._containers import TimeSeriesResult


def phoenix_break(
    y: np.ndarray | list[float],
    *,
    min_segment: int = 10,
    penalty: float = 2.0,
) -> TimeSeriesResult:
    """Detect structural breaks in a time series and characterise recovery.

    Uses a CUSUM-based binary segmentation algorithm to find change points,
    then estimates pre-break mean, post-break trough, and recovery time.

    Parameters
    ----------
    y : array-like
        Univariate time series (1D).
    min_segment : int
        Minimum segment length between breaks.
    penalty : float
        BIC-like penalty multiplier for break detection threshold.

    Returns
    -------
    TimeSeriesResult
        ``values`` holds the detrended residuals, ``extra`` has
        ``breakpoints``, ``segments`` (list of dicts with mean/var),
        ``recovery_times``.
    """
    x = np.asarray(y, dtype=float)
    if x.ndim != 1 or len(x) < 2 * min_segment:
        raise ValueError(f"Need 1D array with at least {2 * min_segment} points")

    n = len(x)
    threshold = penalty * np.log(n)

    def _find_break(arr, start):
        m = len(arr)
        if m < 2 * min_segment:
            return -1
        cumsum = np.cumsum(arr - arr.mean())
        S = np.abs(cumsum)
        S[:min_segment] = 0
        S[-min_segment:] = 0
        idx = int(np.argmax(S))
        if S[idx] > threshold * np.sqrt(m):
            return start + idx
        return -1

    breaks = []

    def _segment(arr, start):
        bp = _find_break(arr, start)
        if bp < 0:
            return
        local = bp - start
        breaks.append(bp)
        if local >= 2 * min_segment:
            _segment(arr[:local], start)
        if len(arr) - local >= 2 * min_segment:
            _segment(arr[local:], bp)

    _segment(x, 0)
    breaks = sorted(set(breaks))

    edges = [0] + breaks + [n]
    segments = []
    for i in range(len(edges) - 1):
        seg = x[edges[i] : edges[i + 1]]
        segments.append(
            {
                "start": edges[i],
                "end": edges[i + 1],
                "mean": float(seg.mean()),
                "std": float(seg.std(ddof=1)) if len(seg) > 1 else 0.0,
                "min": float(seg.min()),
                "max": float(seg.max()),
            }
        )

    recovery_times = []
    for i, bp in enumerate(breaks):
        if i == 0 and bp > min_segment:
            pre_mean = segments[0]["mean"]
        elif i > 0:
            pre_mean = segments[i]["mean"]
        else:
            continue
        post = x[bp:]
        recovered = np.where(post >= pre_mean)[0]
        if len(recovered) > 0:
            recovery_times.append({"breakpoint": bp, "recovery_steps": int(recovered[0])})
        else:
            recovery_times.append({"breakpoint": bp, "recovery_steps": None})

    grand_mean = float(x.mean())
    residuals = x - grand_mean

    return TimeSeriesResult(
        name="phoenix_break",
        values=residuals,
        extra={
            "breakpoints": breaks,
            "segments": segments,
            "recovery_times": recovery_times,
            "n": n,
        },
    )


phoen = phoenix_break


def cheatsheet() -> str:
    return "phoenix_break({}) -> Time series structural break detection + recovery. 'I am fir"
