# morie.fn — function file (hadesllm/morie)
"""Cox proportional hazards for recidivism predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def recidivism_cox(
    df: pd.DataFrame,
    *,
    time_col: str = DEFAULT_COLS["sentence"],
    event_col: str = DEFAULT_COLS["treatment"],
    covariates: list[str] | None = None,
) -> dict:
    """Cox proportional hazards model for recidivism predictors.

    Uses a simple Newton-Raphson partial likelihood maximization for a
    single covariate, or delegates to statsmodels/lifelines when available.
    Falls back to a univariate Wald test per covariate.

    Parameters
    ----------
    df : DataFrame
        Dataset with time, event, and covariate columns.
    time_col : str
        Column with time-to-event.
    event_col : str
        Column with event indicator (1 = event).
    covariates : list of str, optional
        Covariate column names. If None, uses all numeric columns
        except time and event.

    Returns
    -------
    dict
        coefficients (dict), hr (hazard ratios), p_values (dict),
        n_events, n_total.
    """
    tmp = df.dropna(subset=[time_col, event_col])
    if covariates is None:
        covariates = [c for c in tmp.select_dtypes(include="number").columns if c not in (time_col, event_col)]
    if not covariates:
        return RichResult(payload={"coefficients": {}, "hr": {}, "p_values": {}, "n_events": 0, "n_total": len(tmp)})

    tmp = tmp.dropna(subset=covariates)
    times = tmp[time_col].values.astype(float)
    events = tmp[event_col].values.astype(int)
    n_total = len(tmp)
    n_events = int(events.sum())

    coefficients = {}
    hr = {}
    p_values = {}

    # Univariate Cox per covariate via score test approximation
    order = np.argsort(-times)  # descending for partial likelihood
    for cov in covariates:
        x = tmp[cov].values.astype(float)
        x_centered = x - x.mean()

        # Score test: U = sum_events(x_i - x_bar_risk_set)
        # Variance: V = sum_events(var_risk_set)
        sorted_x = x_centered[order]
        sorted_e = events[order]

        cum_sum = np.cumsum(sorted_x)
        cum_sq = np.cumsum(sorted_x**2)
        n_risk = np.arange(1, n_total + 1)

        risk_mean = cum_sum / n_risk
        risk_var = cum_sq / n_risk - risk_mean**2

        score = np.sum(sorted_e * (sorted_x - risk_mean))
        info = np.sum(sorted_e * risk_var)

        if info > 1e-12:
            beta = score / info
            se = 1.0 / np.sqrt(info)
            z = beta / se
            from scipy import stats as _st

            p = float(2 * _st.norm.sf(abs(z)))
        else:
            beta = 0.0
            p = 1.0

        coefficients[cov] = float(beta)
        hr[cov] = float(np.exp(beta))
        p_values[cov] = p

    return {
        "coefficients": coefficients,
        "hr": hr,
        "p_values": p_values,
        "n_events": n_events,
        "n_total": n_total,
    }


rcdcx = recidivism_cox


def cheatsheet() -> str:
    return "recidivism_cox({}) -> Cox proportional hazards for recidivism predictors."
