"""Survival model discrimination measures."""

from __future__ import annotations

import numpy as np

__all__ = ["srmdc"]


def srmdc(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    eval_times: np.ndarray | None = None,
) -> dict:
    """Discrimination measures for survival models.

    Computes C-index, time-dependent AUC, and D-statistic.

    Parameters
    ----------
    predicted_risk : array-like
        Predicted risk scores (higher = worse prognosis).
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event).
    eval_times : array-like, optional
        Times for time-dependent AUC. Default: quartiles.

    Returns
    -------
    dict
        c_index, d_statistic, auc_times, auc_values, n_obs.
    """
    risk = np.asarray(predicted_risk, dtype=float)
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=int)
    n = len(t)

    concordant = 0
    discordant = 0
    tied = 0
    for i in range(n):
        if d[i] == 0:
            continue
        for j in range(n):
            if i == j or (t[j] <= t[i] and d[j] == 0):
                continue
            if t[j] > t[i]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1
    total = concordant + discordant + tied
    c_index = (concordant + 0.5 * tied) / total if total > 0 else 0.5

    risk_sorted = np.sort(risk)
    lo = np.percentile(risk, 25)
    hi = np.percentile(risk, 75)
    d_stat = float(hi - lo) if not np.isnan(hi - lo) else 0.0

    if eval_times is None:
        eval_times = np.percentile(t[d == 1], [25, 50, 75]) if np.sum(d) > 3 else np.array([np.median(t)])
    eval_times = np.asarray(eval_times, dtype=float)

    auc_vals = []
    for et in eval_times:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        med_risk = np.median(risk)
        for i in range(n):
            pred_pos = risk[i] > med_risk
            actual_pos = (t[i] <= et) & (d[i] == 1)
            if pred_pos and actual_pos:
                tp += 1
            elif pred_pos and not actual_pos:
                fp += 1
            elif not pred_pos and actual_pos:
                fn += 1
            else:
                tn += 1
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        auc_vals.append(0.5 * (sens + spec))

    return {
        "c_index": float(c_index),
        "d_statistic": d_stat,
        "auc_times": eval_times,
        "auc_values": np.array(auc_vals),
        "n_obs": n,
    }


srmdc_fn = srmdc


def cheatsheet() -> str:
    return "srmdc(predicted_risk, time, event) -> Survival model discrimination."
