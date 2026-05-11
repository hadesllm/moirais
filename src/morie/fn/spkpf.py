"""Single-peaked preference check. 'Rose Whip!' -- Kurama, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def single_peaked_check(preferences):
    """Check if preference orderings are single-peaked.

    Parameters
    ----------
    preferences : array-like
        Matrix (n_resp x n_stim) of ratings or rankings.

    Returns
    -------
    DescriptiveResult
        value = bool (all single-peaked), extra has per-row results.
    """
    import numpy as np

    P = np.asarray(preferences, dtype=float)
    n_resp, n_stim = P.shape
    results = []
    for i in range(n_resp):
        row = P[i]
        peak_idx = int(np.argmax(row))
        left_ok = all(row[j] <= row[j + 1] for j in range(peak_idx))
        right_ok = all(row[j] >= row[j + 1] for j in range(peak_idx, n_stim - 1))
        results.append(left_ok and right_ok)
    all_sp = all(results)
    return DescriptiveResult(
        name="single_peaked_check",
        value=all_sp,
        extra={"per_row": results, "n_single_peaked": sum(results), "n_resp": n_resp},
    )


spkpf = single_peaked_check


def cheatsheet() -> str:
    return "single_peaked_check({}) -> Single-peaked preference check. 'Rose Whip!' -- Kurama, Yu Y"
