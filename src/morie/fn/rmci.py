# morie.fn — function file (hadesllm/morie)
"""Reliable Change Index (Jacobson & Truax)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn.rsem import rsem


def rmci(
    pre: np.ndarray | pd.Series,
    post: np.ndarray | pd.Series,
    *,
    sem: float | None = None,
    data: pd.DataFrame | np.ndarray | None = None,
    reliability: float | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Reliable Change Index (RCI) per Jacobson & Truax (1991).

    RCI = (X2 - X1) / (SEM * sqrt(2))

    Tests whether an individual's change score exceeds measurement
    error.  |RCI| > z_crit indicates statistically reliable change.

    Parameters
    ----------
    pre : array-like
        Pre-test total scores (one per person).
    post : array-like
        Post-test total scores (same length as ``pre``).
    sem : float or None
        Pre-computed SEM.  If None, computed from ``data``.
    data : DataFrame, ndarray, or None
        Item matrix for computing SEM.  Required if ``sem`` is None.
    reliability : float or None
        Reliability for SEM computation.  If None, uses alpha.
    alpha : float
        Significance level for flagging change (default 0.05).

    Returns
    -------
    DataFrame
        Columns: ``id``, ``pre``, ``post``, ``change``, ``rci``,
        ``significant`` (bool), ``direction`` ('improved', 'declined',
        'unchanged').

    Raises
    ------
    ValueError
        If ``pre`` and ``post`` have different lengths, or neither
        ``sem`` nor ``data`` is provided.

    References
    ----------
    Jacobson, N. S., & Truax, P. (1991). Clinical significance: A
    statistical approach to defining meaningful change in
    psychotherapy research. *Journal of Consulting and Clinical
    Psychology*, 59(1), 12-19.
    """
    pre_arr = np.asarray(pre, dtype=np.float64).ravel()
    post_arr = np.asarray(post, dtype=np.float64).ravel()

    if len(pre_arr) != len(post_arr):
        raise ValueError(f"pre and post must have same length, got {len(pre_arr)} and {len(post_arr)}")

    if sem is None:
        if data is None:
            raise ValueError("Must provide either 'sem' or 'data'.")
        sem = rsem(data, reliability=reliability)

    if np.isnan(sem) or sem < 1e-15:
        # Cannot compute RCI without valid SEM
        n = len(pre_arr)
        return pd.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "pre": pre_arr,
                "post": post_arr,
                "change": post_arr - pre_arr,
                "rci": np.full(n, np.nan),
                "significant": np.full(n, False),
                "direction": ["unchanged"] * n,
            }
        )

    from scipy import stats as sp

    z_crit = sp.norm.ppf(1.0 - alpha / 2.0)

    s_diff = sem * np.sqrt(2.0)
    change = post_arr - pre_arr
    rci = change / s_diff

    significant = np.abs(rci) > z_crit
    direction = np.where(~significant, "unchanged", np.where(change > 0, "improved", "declined"))

    return pd.DataFrame(
        {
            "id": np.arange(1, len(pre_arr) + 1),
            "pre": pre_arr,
            "post": post_arr,
            "change": change,
            "rci": rci,
            "significant": significant,
            "direction": direction,
        }
    )


short = rmci


def cheatsheet() -> str:
    return "rmci({}) -> Reliable Change Index (Jacobson & Truax)."
