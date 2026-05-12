# morie.fn -- function file (hadesllm/morie)
"""Pseudo-guessing parameter analysis for 3PL IRT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def irt_guessing(
    item_params: dict,
    *,
    n_options: int = 4,
) -> DescriptiveResult:
    """Analyse pseudo-guessing (c) parameters from a 3PL model.

    Compares estimated c to the chance level (1/n_options).

    Parameters
    ----------
    item_params : dict
        {item_name: {"c": float, ...}} for each item.
    n_options : int
        Number of MC options for chance-level comparison (default 4).

    Returns
    -------
    DescriptiveResult
        value=DataFrame with item, guessing param, above/below chance.

    References
    ----------
    Barton, M. A. & Lord, F. M. (1981). An upper asymptote for the
    three-parameter logistic item-response model. ETS Research Report.
    """
    chance = 1.0 / n_options
    rows = []
    for name, params in item_params.items():
        c = params.get("c", 0.0)
        rows.append(
            {
                "item": name,
                "c_param": float(c),
                "chance_level": chance,
                "above_chance": c > chance,
            }
        )

    df = pd.DataFrame(rows)
    c_vals = np.array([r["c_param"] for r in rows])

    return DescriptiveResult(
        name="IRT guessing",
        value=df,
        extra={
            "mean_c": float(np.mean(c_vals)),
            "n_above_chance": int(sum(r["above_chance"] for r in rows)),
            "chance_level": chance,
        },
    )


guessing = irt_guessing


def cheatsheet() -> str:
    return "irt_guessing({}) -> Pseudo-guessing parameter analysis for 3PL IRT."
