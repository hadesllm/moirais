# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier survival estimator with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def kmsurv(times: Union[Sequence, np.ndarray],
           events: Union[Sequence, np.ndarray]):
    """Kaplan-Meier estimator of S(t) for right-censored data."""
    from ._richresult import RichResult
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    if t.shape != e.shape:
        raise ValueError(f"times/events shape mismatch: {t.shape} vs {e.shape}.")
    if np.any(t < 0):
        raise ValueError("times must be non-negative.")
    order = np.argsort(t)
    t = t[order]; e = e[order]
    unique_t = np.unique(t)
    n = t.size
    surv = 1.0
    rows = []
    out_t, out_s = [], []
    for ut in unique_t:
        at_risk = int(np.sum(t >= ut))
        d_k = int(np.sum((t == ut) & (e == 1)))
        c_k = int(np.sum((t == ut) & (e == 0)))
        if at_risk > 0:
            surv *= (1 - d_k / at_risk)
        rows.append([f"{ut:g}", at_risk, d_k, c_k, f"{surv:.4f}"])
        out_t.append(float(ut))
        out_s.append(float(surv))
    censored_total = int(np.sum(e == 0))
    events_total = int(np.sum(e == 1))
    median_idx = next((i for i, s in enumerate(out_s) if s <= 0.5), None)
    median_time = out_t[median_idx] if median_idx is not None else None
    warnings = []
    if censored_total / n > 0.5:
        warnings.append(f"{censored_total}/{n} ({100*censored_total/n:.0f}%) "
                        "censored - tail estimates unreliable.")
    return RichResult(
        title="Kaplan-Meier survival estimator",
        summary_lines=[("n total", n), ("Events", events_total),
                       ("Censored", censored_total),
                       ("Median survival time", median_time if median_time else "not reached"),
                       ("S(end-of-followup)", out_s[-1] if out_s else float('nan'))],
        tables=[{"title": "Step-function table:",
                 "headers": ["t", "at risk", "events", "censored", "S(t)"],
                 "rows": rows}],
        warnings=warnings,
        payload={"times": out_t, "survival": out_s,
                 "events": events_total, "censored": censored_total},
    )
