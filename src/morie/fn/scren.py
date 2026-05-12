# morie.fn -- function file (hadesllm/morie)
"""Screening test performance (Se, Sp, PPV, NPV, LR+, LR-)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def screening_performance(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    *,
    prevalence: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute screening/diagnostic test performance metrics.

    From a 2x2 confusion matrix:

    .. math::

        Se = \\frac{TP}{TP + FN}, \\quad Sp = \\frac{TN}{TN + FP}

        PPV = \\frac{TP}{TP + FP}, \\quad NPV = \\frac{TN}{TN + FN}

        LR^+ = \\frac{Se}{1 - Sp}, \\quad LR^- = \\frac{1 - Se}{Sp}

    Parameters
    ----------
    tp, fp, fn, tn : int
        Counts for true positive, false positive, false negative,
        true negative.
    prevalence : float or None
        If provided, PPV/NPV are computed for this prevalence
        (Bayesian adjustment) rather than from the table directly.
    alpha : float, default 0.05
        Significance level for Wilson score CIs on Se and Sp.

    Returns
    -------
    dict
        Keys: 'sensitivity', 'specificity', 'ppv', 'npv',
              'lr_positive', 'lr_negative', 'accuracy',
              'se_ci', 'sp_ci', 'youden_index'.

    References
    ----------
    Altman, D. G. & Bland, J. M. (1994). Diagnostic tests. Statistics
    notes. BMJ, 309(6947), 102.
    """
    if any(x < 0 for x in (tp, fp, fn, tn)):
        raise ValueError("All cell counts must be non-negative.")

    pos = tp + fn
    neg = tn + fp
    total = tp + fp + fn + tn

    se = tp / pos if pos > 0 else 0.0
    sp = tn / neg if neg > 0 else 0.0

    if prevalence is not None:
        if not (0 <= prevalence <= 1):
            raise ValueError("prevalence must be in [0, 1].")
        ppv_num = se * prevalence
        ppv_den = ppv_num + (1 - sp) * (1 - prevalence)
        ppv = ppv_num / ppv_den if ppv_den > 0 else 0.0

        npv_num = sp * (1 - prevalence)
        npv_den = npv_num + (1 - se) * prevalence
        npv = npv_num / npv_den if npv_den > 0 else 0.0
    else:
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    lr_pos = se / (1 - sp) if sp < 1.0 else np.inf
    lr_neg = (1 - se) / sp if sp > 0 else np.inf

    accuracy = (tp + tn) / total if total > 0 else 0.0
    youden = se + sp - 1.0

    def _wilson(x, n):
        if n == 0:
            return (0.0, 1.0)
        p = x / n
        z = _st.norm.ppf(1 - alpha / 2)
        d = 1 + z**2 / n
        c = p + z**2 / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))

    se_ci = _wilson(tp, pos)
    sp_ci = _wilson(tn, neg)

    return {
        "sensitivity": float(se),
        "specificity": float(sp),
        "ppv": float(ppv),
        "npv": float(npv),
        "lr_positive": float(lr_pos),
        "lr_negative": float(lr_neg),
        "accuracy": float(accuracy),
        "youden_index": float(youden),
        "se_ci": se_ci,
        "sp_ci": sp_ci,
    }


scren = screening_performance


def cheatsheet() -> str:
    return "screening_performance({}) -> Se, Sp, PPV, NPV, LR+, LR-."
