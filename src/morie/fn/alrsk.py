# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Composite alert risk score."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alrisk(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
    weight_mh: float = 1.0,
    weight_sr: float = 1.0,
    weight_sw: float = 1.0,
) -> pd.DataFrame:
    """Composite alert risk score.

    Computes a weighted sum of active alerts (0--3) per row.
    Default weights are all 1.0 (unweighted sum). Users can assign
    higher weights to more clinically severe alerts.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Unique individual identifier column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.
    weight_mh, weight_sr, weight_sw : float
        Weights for each alert type.

    Returns
    -------
    DataFrame
        Original data with added ``risk_score`` column.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    weights = [weight_mh, weight_sr, weight_sw]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["risk_score"] = sum(data[c] * w for c, w in zip(acols, weights))
    return data


short = alrisk


def cheatsheet() -> str:
    return "alrisk({}) -> Composite alert risk score."
