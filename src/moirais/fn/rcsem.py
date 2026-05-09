# moirais.fn — function file (hadesllm/moirais)
"""Conditional Standard Error of Measurement by score level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rcsem(
    data: pd.DataFrame | np.ndarray,
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Conditional SEM by total score quintile.

    SEM varies by score level: items near extremes of the score range
    tend to have lower variance, producing non-uniform measurement
    precision.  This function bins respondents by total score and
    computes the SEM within each bin, giving a score-conditional view
    of measurement error.

    Within each bin, CSEM is the SD of item-level residuals (observed
    minus bin-mean) pooled across items.  Formally:
        CSEM_b = sqrt(mean of item variances within bin b)

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).
    n_bins : int
        Number of score bins (default 5 for quintiles).

    Returns
    -------
    DataFrame
        Columns: ``score_bin`` (label), ``score_lo``, ``score_hi``,
        ``n`` (count), ``mean_score``, ``sem``.

    References
    ----------
    Feldt, L. S., Steffen, M., & Gupta, N. C. (1985). A comparison of
    five methods for estimating the standard error of measurement at
    specific score levels. *Applied Psychological Measurement*, 9(4),
    351-361.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2 or n < n_bins:
        return pd.DataFrame(columns=["score_bin", "score_lo", "score_hi", "n", "mean_score", "sem"])

    totals = X.sum(axis=1)

    # Use quantile-based bins to ensure roughly equal n per bin
    try:
        bin_labels = pd.qcut(totals, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Too few unique values for requested bins
        bin_labels = pd.cut(totals, bins=n_bins, labels=False)
        bin_labels = np.asarray(bin_labels)

    bin_labels = np.asarray(bin_labels, dtype=np.float64)
    unique_bins = np.unique(bin_labels[np.isfinite(bin_labels)])

    rows = []
    for b_idx, b in enumerate(unique_bins):
        in_bin = bin_labels == b
        X_bin = X[in_bin]
        n_bin = X_bin.shape[0]

        if n_bin < 2:
            continue

        totals_bin = X_bin.sum(axis=1)

        # Item variances within this score bin
        item_vars = np.var(X_bin, axis=0, ddof=1)
        csem = np.sqrt(item_vars.mean())

        rows.append(
            {
                "score_bin": f"Q{b_idx + 1}",
                "score_lo": float(totals_bin.min()),
                "score_hi": float(totals_bin.max()),
                "n": int(n_bin),
                "mean_score": float(totals_bin.mean()),
                "sem": float(csem),
            }
        )

    return pd.DataFrame(rows)


short = rcsem


def cheatsheet() -> str:
    return "rcsem({}) -> Conditional Standard Error of Measurement by score level."
