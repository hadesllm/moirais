# moirais.fn — function file (hadesllm/moirais)
"""Diagnostic Odds Ratio (DOR)."""

from __future__ import annotations

import math
from typing import Any


def diagnostic_odds_ratio(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, Any]:
    """Compute the Diagnostic Odds Ratio with 95% CI.

    DOR = (TP * TN) / (FP * FN)

    The 95% CI is computed on the log scale:
    log(DOR) +/- 1.96 * SE(log(DOR))
    where SE = sqrt(1/TP + 1/FP + 1/FN + 1/TN).

    Parameters
    ----------
    tp : int
        True positives.
    fp : int
        False positives.
    fn : int
        False negatives.
    tn : int
        True negatives.

    Returns
    -------
    dict
        dor, ci_lower, ci_upper, log_dor, se_log_dor.

    Raises
    ------
    ValueError
        If any cell is negative.

    References
    ----------
    Glas, A. S., Lijmer, J. G., Prins, M. H., Bonsel, G. J., &
        Bossuyt, P. M. M. (2003). The diagnostic odds ratio: a single
        indicator of test performance. *Journal of Clinical Epidemiology*,
        56(11), 1129-1135. doi:10.1016/S0895-4356(03)00177-X
    """
    for name, val in [("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}.")

    # Add 0.5 continuity correction if any cell is 0
    a, b, c, d = float(tp), float(fp), float(fn), float(tn)
    if any(v == 0 for v in [a, b, c, d]):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    dor_val = (a * d) / (b * c)
    log_dor = math.log(dor_val)
    se_log = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)

    ci_lower = math.exp(log_dor - 1.96 * se_log)
    ci_upper = math.exp(log_dor + 1.96 * se_log)

    return {
        "dor": dor_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "log_dor": log_dor,
        "se_log_dor": se_log,
    }


dor = diagnostic_odds_ratio


def cheatsheet() -> str:
    return "diagnostic_odds_ratio({}) -> Diagnostic Odds Ratio (DOR)."
