# morie.fn — function file (hadesllm/morie)
"""Extract IRT discrimination parameters."""

from __future__ import annotations

import pandas as pd


def irt_discrimination(item_params: dict) -> pd.DataFrame:
    """Extract discrimination (a) parameters from IRT item parameter dict.

    Parameters
    ----------
    item_params : dict
        {item_name: {'a': ..., 'b': ..., ...}}.

    Returns
    -------
    DataFrame
        Columns: item, a (discrimination).

    References
    ----------
    Baker, F. B. & Kim, S. H. (2004). Item Response Theory: Parameter
    Estimation Techniques (2nd ed.). Marcel Dekker.
    """
    rows = []
    for item, params in item_params.items():
        a = params.get("a", 1.0)  # Rasch model default a=1
        rows.append({"item": str(item), "a": float(a)})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "irt_discrimination({}) -> Extract IRT discrimination parameters."
