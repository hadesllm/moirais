# moirais.fn — function file (hadesllm/moirais)
"""IRT distractor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irt_distractor(
    data: pd.DataFrame | np.ndarray,
    *,
    key: dict[str, int] | None = None,
    n_groups: int = 3,
) -> dict:
    """Distractor analysis: proportion choosing each option by score group.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    key : dict or None
        Answer key {item_name: correct_option}. If None, analysis is
        descriptive only (no correct/incorrect tagging).
    n_groups : int
        Number of score groups (default 3: low/medium/high).

    Returns
    -------
    dict
        {item_name: DataFrame with columns option, group_1, group_2, ...
        and optionally 'correct' flag}.

    References
    ----------
    Haladyna, T. M. & Rodriguez, M. C. (2013). Developing and Validating
    Test Items. Routledge.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    # Score each person (using key if provided, else sum)
    if key is not None:
        scores = np.zeros(len(data))
        for item, correct in key.items():
            if item in data.columns:
                scores += (data[item] == correct).astype(float).values
    else:
        numeric = data.select_dtypes(include=[np.number])
        scores = numeric.sum(axis=1).values

    # Create score groups
    try:
        groups = pd.qcut(scores, n_groups, labels=False, duplicates="drop") + 1
    except ValueError:
        groups = np.ones(len(data), dtype=int)

    result = {}
    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.number):
            continue
        options = sorted(data[col].dropna().unique())
        rows = []
        for opt in options:
            row = {"option": opt}
            for g in sorted(np.unique(groups)):
                mask = groups == g
                n_in_group = mask.sum()
                if n_in_group > 0:
                    row[f"group_{int(g)}"] = float(np.mean(data.loc[mask, col] == opt))
                else:
                    row[f"group_{int(g)}"] = 0.0
            if key is not None and col in key:
                row["correct"] = opt == key[col]
            rows.append(row)
        result[col] = pd.DataFrame(rows)

    return result


def cheatsheet() -> str:
    return "irt_distractor({}) -> IRT distractor analysis."
