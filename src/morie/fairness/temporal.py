# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-city temporal disparity audit.

Reimplements the longitudinal, multi-city bias audit of Barman &
Barman, *Unmasking Algorithmic Bias in Predictive Policing*
(arXiv:2603.18987): the four disparity metrics — Disparate Impact
Ratio, Demographic Parity Gap, Gini coefficient, and Bias
Amplification Score — are computed for each ``(city, time-period)``
cell and assembled into a time series, so that *temporal instability*
and *cross-city divergence* become visible.

The paper's central auditing lesson is that bias metrics are **not**
stable from one deployment cycle to the next — Baltimore's annual mean
DIR swings from 0.08 (2018) to 15,714 (2019) — and therefore must be
recomputed per period and per city rather than measured once at
deployment.  :func:`predpol_temporal_audit` makes that instability a
first-class output.

Builds on the Phase A metrics in :mod:`morie.fairness.metrics`;
methods reimplemented from the paper, no code copied.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from morie.fairness.metrics import (
    _ordered_unique,
    fairness_bias_amplification,
    fairness_demographic_parity,
    fairness_disparate_impact,
    fairness_gini,
)
from morie.fn._richresult import RichResult

__all__ = ["predpol_temporal_audit"]


def _mean(xs: list[float]) -> float:
    finite = [x for x in xs if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def predpol_temporal_audit(
    period: Any,
    city: Any,
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Audit how disparity metrics move over time and across cities.

    For every ``(city, period)`` cell the audit computes the four
    disparity metrics, then aggregates them per city — reporting, for
    each city, the mean of each metric, the count of periods with
    DIR > 1 (over-prediction months), and the **DIR temporal range**
    (max − min), which quantifies how unstable the metric is across the
    audited window.

    Parameters
    ----------
    period : array-like
        Time-period label for each record (e.g. ``"2019-03"``).  Sorted
        lexically for display, so ISO-style labels order correctly.
    city : array-like
        City label for each record.
    y_pred : array-like
        The decision/assignment for each record.
    group : array-like
        Protected attribute for each record.
    privileged : optional
        Reference group.  If omitted it is inferred **globally** (from
        the pooled data) so every cell uses the same reference.
    favorable : default ``1``
        The value of ``y_pred`` counted as the favourable outcome.

    Returns
    -------
    RichResult
        Headline value is the largest per-city DIR temporal range — the
        worst temporal instability found.

    Examples
    --------
    >>> import morie
    >>> # one city, two periods: identical disparity each period
    >>> period = ["p1"] * 10 + ["p2"] * 10
    >>> city   = ["A"] * 20
    >>> pred   = ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]) * 2
    >>> grp    = (["X"] * 5 + ["Y"] * 5) * 2
    >>> res = morie.predpol_temporal_audit(period, city, pred, grp,
    ...                                    privileged="X")
    >>> round(res.payload["per_city"]["A"]["dir_range"], 6)  # stable
    0.0
    """
    period = np.asarray(list(period), dtype=object)
    city = np.asarray(list(city), dtype=object)
    y_pred = np.asarray(list(y_pred))
    group = np.asarray(list(group), dtype=object)
    n = len(period)
    if not (n == len(city) == len(y_pred) == len(group)):
        raise ValueError("period, city, y_pred and group must all align")
    if n == 0:
        raise ValueError("inputs are empty")

    warnings: list[str] = []
    if privileged is None:
        pooled: dict[Any, float] = {}
        for g in _ordered_unique(group):
            m = group == g
            pooled[g] = float(np.mean(y_pred[m] == favorable))
        privileged = max(pooled, key=lambda k: pooled[k])
        warnings.append(
            f"`privileged` not given; inferred globally as {privileged!r} "
            f"(highest pooled favourable-outcome rate) so every cell uses "
            f"the same reference group."
        )

    cells: list[dict] = []
    skipped = 0
    for c in _ordered_unique(city):
        city_mask = city == c
        for p in sorted(_ordered_unique(period[city_mask]), key=str):
            mask = city_mask & (period == p)
            cg = group[mask]
            cy = y_pred[mask]
            cell_groups = _ordered_unique(cg)
            if len(cell_groups) < 2 or privileged not in cell_groups:
                skipped += 1
                continue
            di = float(fairness_disparate_impact(
                cy, cg, privileged=privileged, favorable=favorable))
            pg = float(fairness_demographic_parity(
                cy, cg, privileged=privileged, favorable=favorable))
            rate_vec = [
                float(np.mean(cy[cg == g] == favorable)) for g in cell_groups
            ]
            gini = float(fairness_gini(rate_vec))
            bas = float(fairness_bias_amplification(
                cy, cg, privileged=privileged, favorable=favorable))
            cells.append({
                "city": c, "period": p, "n": int(mask.sum()),
                "dir": di, "parity_gap": pg, "gini": gini, "bas": bas,
            })

    if skipped:
        warnings.append(
            f"{skipped} (city, period) cell(s) were skipped — fewer than "
            f"two groups present, or the privileged group absent."
        )
    if not cells:
        raise ValueError(
            "no (city, period) cell had enough groups to audit"
        )

    per_city: dict[Any, dict] = {}
    for c in _ordered_unique(np.array([x["city"] for x in cells],
                                      dtype=object)):
        cc = [x for x in cells if x["city"] == c]
        dirs = [x["dir"] for x in cc if np.isfinite(x["dir"])]
        per_city[c] = {
            "n_periods": len(cc),
            "mean_dir": _mean([x["dir"] for x in cc]),
            "mean_parity_gap": _mean([x["parity_gap"] for x in cc]),
            "mean_gini": _mean([x["gini"] for x in cc]),
            "mean_bas": _mean([x["bas"] for x in cc]),
            "dir_min": min(dirs) if dirs else float("nan"),
            "dir_max": max(dirs) if dirs else float("nan"),
            "dir_range": (max(dirs) - min(dirs)) if dirs else float("nan"),
            "periods_dir_gt1": sum(1 for d in dirs if d > 1.0),
        }

    ranges = [v["dir_range"] for v in per_city.values()
              if np.isfinite(v["dir_range"])]
    worst_range = max(ranges) if ranges else float("nan")
    mean_dirs = [v["mean_dir"] for v in per_city.values()
                 if np.isfinite(v["mean_dir"])]
    cross_city_spread = (max(mean_dirs) - min(mean_dirs)
                         if len(mean_dirs) >= 2 else 0.0)

    cell_rows = [
        [str(x["city"]), str(x["period"]), x["n"],
         round(x["dir"], 4), round(x["parity_gap"], 4),
         round(x["gini"], 4), round(x["bas"], 4)]
        for x in cells
    ]
    city_rows = [
        [str(c), v["n_periods"], round(v["mean_dir"], 4),
         round(v["mean_parity_gap"], 4), round(v["mean_gini"], 4),
         round(v["mean_bas"], 4), round(v["dir_range"], 4),
         v["periods_dir_gt1"]]
        for c, v in per_city.items()
    ]

    if np.isfinite(worst_range) and worst_range >= 0.5:
        stab = (f"Bias is temporally unstable: the Disparate Impact Ratio "
                f"swings by up to {worst_range:.3f} across periods within "
                f"a single city. A one-off audit at deployment time would "
                f"not have caught this — the metric must be recomputed "
                f"every period.")
    else:
        stab = ("The Disparate Impact Ratio is reasonably stable across "
                "periods within each city over the audited window.")
    if len(per_city) >= 2 and cross_city_spread >= 0.3:
        div = (f" Bias also diverges across cities: mean annual DIR spans "
               f"{cross_city_spread:.3f} between cities — the direction and "
               f"size of bias is city-specific, not a fixed property of "
               f"the system.")
    else:
        div = ""

    return RichResult(
        title="Multi-City Temporal Disparity Audit",
        summary_lines=[
            ("Cells audited", len(cells)),
            ("Cities", len(per_city)),
            ("Worst DIR temporal range", worst_range),
            ("Cross-city mean-DIR spread", cross_city_spread),
            ("Reference group", privileged),
        ],
        sections=[{
            "title": "Per-city aggregates:",
            "headers": ["city", "periods", "mean DIR", "mean PG",
                        "mean Gini", "mean BAS", "DIR range", "M>1"],
            "table": city_rows,
        }],
        tables=[{
            "title": "Per-(city, period) metrics:",
            "headers": ["city", "period", "n", "DIR", "parity gap",
                        "Gini", "BAS"],
            "rows": cell_rows,
        }],
        warnings=warnings,
        interpretation=stab + div,
        payload={
            "value": worst_range,
            "worst_dir_range": worst_range,
            "cross_city_dir_spread": cross_city_spread,
            "per_city": per_city,
            "cells": cells,
            "privileged": privileged,
        },
    )
