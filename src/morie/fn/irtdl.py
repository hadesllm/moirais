# morie.fn -- function file (hadesllm/morie)
"""Extract IRT difficulty parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irt_difficulty(item_params: dict) -> pd.DataFrame:
    """Extract difficulty (b) parameters from IRT item parameter dict.

    Parameters
    ----------
    item_params : dict
        {item_name: {'a': ..., 'b': ..., 'c': ...}} or
        {item_name: {'b': ...}}.

    Returns
    -------
    DataFrame
        Columns: item, b (difficulty).

    References
    ----------
    Embretson, S. E. & Reise, S. P. (2000). Item Response Theory for
    Psychologists. Lawrence Erlbaum.
    """
    rows = []
    for item, params in item_params.items():
        b = params.get("b", np.nan)
        if isinstance(b, (list, np.ndarray)):
            # GRM: multiple thresholds
            for idx, bval in enumerate(b):
                rows.append({"item": f"{item}_t{idx + 1}", "b": float(bval)})
        else:
            rows.append({"item": str(item), "b": float(b)})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "irt_difficulty({}) -> Extract IRT difficulty parameters."
