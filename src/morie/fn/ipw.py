# morie.fn — function file (hadesllm/morie)
"""
Inverse Probability of Treatment Weighting (IPTW).

Implements ``calculate_ipw_weights`` — computes raw or stabilised IPW weights
from propensity scores, with optional quantile trimming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_ipw_weights(
    data: pd.DataFrame,
    treatment: str,
    ps_col: str,
    *,
    stabilized: bool = False,
    trim_quantiles: tuple[float, float] | None = None,
) -> pd.Series:
    """
    Calculate inverse probability of treatment weights (IPTW).

    :param data: The dataframe containing treatment assignment and propensity scores.
    :type data: pandas.DataFrame
    :param treatment: The column name for the actual treatment assignment.
    :type treatment: str
    :param ps_col: The column name containing the propensity scores.
    :type ps_col: str
    :param stabilized: If True, compute stabilised weights using the marginal
        treatment probability.  Default False.
    :type stabilized: bool
    :param trim_quantiles: If provided, a tuple ``(lower, upper)`` of quantiles
        at which to clip the weights (e.g. ``(0.01, 0.99)``).
    :type trim_quantiles: tuple[float, float] or None
    :return: A pandas Series containing the IPTW for each observation.
    :rtype: pandas.Series
    """
    ps = data[ps_col].clip(lower=0.01, upper=0.99)
    t = data[treatment]
    weights = (t / ps) + ((1 - t) / (1 - ps))
    if stabilized:
        p_treated = float(t.mean())
        weights = np.where(t == 1, p_treated / ps, (1 - p_treated) / (1 - ps))
        weights = pd.Series(weights, index=data.index, name="ipw")
    else:
        weights = pd.Series(weights, index=data.index, name="ipw")

    if trim_quantiles is not None:
        lower, upper = trim_quantiles
        q_lower = float(weights.quantile(lower))
        q_upper = float(weights.quantile(upper))
        weights = weights.clip(lower=q_lower, upper=q_upper)
    return weights


ipw_weights = calculate_ipw_weights


def cheatsheet() -> str:
    return "calculate_ipw_weights({}) -> Inverse Probability of Treatment Weighting (IPTW)."
