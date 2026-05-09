# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Concordance index (C-statistic). 'Everything flows. — Heraclitus'

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def concordance_index(y_true, y_score) -> ESRes:
    """Harrell's C-index: proportion of concordant pairs.

    Parameters
    ----------
    y_true : array-like
        True outcome values.
    y_score : array-like
        Predicted scores / risk scores.

    Returns
    -------
    ESRes
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] > y_true[j] and y_score[i] > y_score[j]) or (
                    y_true[i] < y_true[j] and y_score[i] < y_score[j]
                ):
                    concordant += 1
                elif y_score[i] != y_score[j]:
                    discordant += 1
    total = concordant + discordant
    c = concordant / total if total > 0 else 0.5
    # SE approximation (Hanley-McNeil)
    se = np.sqrt(c * (1 - c) / max(total, 1)) if total > 0 else 0.0
    z = stats.norm.ppf(0.975)
    return ESRes(
        measure="C-index",
        estimate=float(c),
        ci_lower=max(0, c - z * se),
        ci_upper=min(1, c + z * se),
        se=float(se),
        n=n,
        extra={"concordant": concordant, "discordant": discordant},
    )


c3po = concordance_index


def cheatsheet() -> str:
    return "concordance_index({}) -> Concordance index (C-statistic). 'The odds of navigating an "
