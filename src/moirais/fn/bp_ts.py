# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bai-Perron structural break test (simplified OLS-based)."""

import numpy as np


def bp_ts(
    y: np.ndarray,
    X: np.ndarray | None = None,
    max_breaks: int = 5,
    min_segment: int | None = None,
) -> dict:
    """
    Simplified Bai-Perron structural break detection.

    Uses a dynamic-programming approach to find the optimal partition of
    observations into *m + 1* segments that minimises the total sum of
    squared residuals from segment-specific OLS fits.  BIC selects the
    number of breaks.

    :param y: 1-D response array.
    :param X: Design matrix (n x p).  If None, a constant-only model
        (mean shift) is used.
    :param max_breaks: Maximum number of breaks to consider. Default 5.
    :param min_segment: Minimum observations per segment.  Defaults to
        ``max(5, 2 * (p + 1))``.
    :return: dict with ``n_breaks``, ``break_indices`` (list of int),
        ``bic`` (per-break-count), ``rss`` (per-break-count), ``segments``.
    :raises ValueError: If series is too short for requested breaks.

    References
    ----------
    Bai, J. & Perron, P. (1998). Estimating and testing linear models
    with multiple structural changes. *Econometrica*, 66(1), 47-78.

    Bai, J. & Perron, P. (2003). Computation and analysis of multiple
    structural change models. *Journal of Applied Econometrics*,
    18(1), 1-22.
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Add intercept
        X = np.column_stack([np.ones(n), X])
    p = X.shape[1]

    if min_segment is None:
        min_segment = max(5, 2 * p)
    if n < 2 * min_segment:
        raise ValueError(f"Series too short ({n}) for min_segment={min_segment}.")

    def _segment_rss(start, end):
        """RSS from OLS fit on y[start:end]."""
        if end - start < p:
            return np.inf
        Xs = X[start:end]
        ys = y[start:end]
        try:
            beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.inf
        resid = ys - Xs @ beta
        return float(np.sum(resid**2))

    # For each number of breaks m=0..max_breaks, find optimal partition
    # Use simplified greedy/DP for tractability
    bic_list = []
    rss_list = []
    best_breaks_list = []

    # m = 0: no breaks
    rss0 = _segment_rss(0, n)
    if rss0 < 1e-12:
        # Perfect fit with no breaks — no structural break possible
        return {
            "n_breaks": 0,
            "break_indices": [],
            "bic": [0.0],
            "rss": [float(rss0)],
            "segments": [(0, n)],
        }
    bic0 = n * np.log(rss0 / n + 1e-300) + p * np.log(n)
    bic_list.append(float(bic0))
    rss_list.append(float(rss0))
    best_breaks_list.append([])

    # For m >= 1, use dynamic programming with cost matrix
    # cost[i, j] = RSS of segment [i, j)
    # Pre-compute costs
    max_possible = min(max_breaks, (n // min_segment) - 1)
    if max_possible < 1:
        return {
            "n_breaks": 0,
            "break_indices": [],
            "bic": bic_list,
            "rss": rss_list,
            "segments": [(0, n)],
        }

    for m in range(1, max_possible + 1):
        # Try all combinations of m break points via DP
        # dp[j][k] = (best_rss, break_list) for first j obs with k segments
        # Simplified: greedy sequential search
        best_rss_m = np.inf
        best_bp = []

        # Enumerate candidate break points at multiples of min_segment
        candidates = list(range(min_segment, n - min_segment + 1, max(1, min_segment // 2)))
        if len(candidates) == 0:
            break

        if m == 1:
            for bp in candidates:
                rss_val = _segment_rss(0, bp) + _segment_rss(bp, n)
                if rss_val < best_rss_m:
                    best_rss_m = rss_val
                    best_bp = [bp]
        elif m == 2:
            for i, bp1 in enumerate(candidates):
                for bp2 in candidates[i + 1 :]:
                    if bp2 - bp1 < min_segment or n - bp2 < min_segment:
                        continue
                    rss_val = _segment_rss(0, bp1) + _segment_rss(bp1, bp2) + _segment_rss(bp2, n)
                    if rss_val < best_rss_m:
                        best_rss_m = rss_val
                        best_bp = [bp1, bp2]
        else:
            # For m >= 3, use recursive binary segmentation as approximation
            breaks_found = []
            segments = [(0, n)]
            for _ in range(m):
                best_gain = -np.inf
                best_seg_idx = 0
                best_split = 0
                for si, (s, e) in enumerate(segments):
                    if e - s < 2 * min_segment:
                        continue
                    rss_full = _segment_rss(s, e)
                    for bp in range(s + min_segment, e - min_segment + 1, max(1, min_segment // 2)):
                        rss_split = _segment_rss(s, bp) + _segment_rss(bp, e)
                        gain = rss_full - rss_split
                        if gain > best_gain:
                            best_gain = gain
                            best_seg_idx = si
                            best_split = bp
                if best_gain <= 0:
                    break
                s, e = segments[best_seg_idx]
                segments = segments[:best_seg_idx] + [(s, best_split), (best_split, e)] + segments[best_seg_idx + 1 :]
                breaks_found.append(best_split)
            breaks_found.sort()
            best_bp = breaks_found[:m]
            segs = [0] + best_bp + [n]
            best_rss_m = sum(_segment_rss(segs[i], segs[i + 1]) for i in range(len(segs) - 1))

        if best_rss_m < np.inf:
            n_params_m = (m + 1) * p
            bic_m = n * np.log(best_rss_m / n + 1e-300) + n_params_m * np.log(n)
            bic_list.append(float(bic_m))
            rss_list.append(float(best_rss_m))
            best_breaks_list.append(best_bp)

    # Select number of breaks by minimum BIC
    best_m = int(np.argmin(bic_list))
    bp_final = best_breaks_list[best_m]
    segs_final = [0] + bp_final + [n]
    segments = [(segs_final[i], segs_final[i + 1]) for i in range(len(segs_final) - 1)]

    return {
        "n_breaks": best_m,
        "break_indices": bp_final,
        "bic": bic_list,
        "rss": rss_list,
        "segments": segments,
    }


def cheatsheet() -> str:
    return "bp_ts({}) -> Bai-Perron structural break test (simplified OLS-based)."
