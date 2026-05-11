# morie.fn — function file (hadesllm/morie)
"""Rosenbaum sensitivity bounds for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_sensitivity(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", gamma_range: list[float] | None = None) -> dict:
    """Rosenbaum sensitivity analysis for hidden bias.

    Computes Wilcoxon signed-rank bounds under varying levels of
    unmeasured confounding (Gamma). Approximates paired data by
    matching treated/control on index order.

    Parameters
    ----------
    df : DataFrame
        Data with outcome and treatment columns.
    outcome, treatment : str
        Column names.
    gamma_range : list of float, optional
        Values of Gamma to evaluate. Defaults to [1.0, 1.5, 2.0, 2.5, 3.0].

    Returns
    -------
    dict
        Keys: gamma_values (list), p_upper (list), p_lower (list), n_pairs.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    data = df[[outcome, treatment]].dropna()
    treated = data.loc[data[treatment] == 1, outcome].values
    control = data.loc[data[treatment] == 0, outcome].values

    # Match by index order (simple approximation)
    n_pairs = min(len(treated), len(control))
    if n_pairs < 2:
        return {
            "gamma_values": gamma_range,
            "p_upper": [np.nan] * len(gamma_range),
            "p_lower": [np.nan] * len(gamma_range),
            "n_pairs": 0,
        }

    diffs = treated[:n_pairs] - control[:n_pairs]
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    signs = np.sign(diffs)

    T_obs = float(np.sum(ranks[signs > 0]))

    p_upper = []
    p_lower = []

    for gamma in gamma_range:
        # Under Gamma, probability of positive sign in [1/(1+Gamma), Gamma/(1+Gamma)]
        p_pos = gamma / (1 + gamma)
        p_neg = 1 / (1 + gamma)

        # Expected value and variance of T+ under the bounds
        E_upper = np.sum(ranks * p_pos)
        V_upper = np.sum(ranks**2 * p_pos * (1 - p_pos))
        E_lower = np.sum(ranks * p_neg)
        V_lower = np.sum(ranks**2 * p_neg * (1 - p_neg))

        z_upper = (T_obs - E_upper) / np.sqrt(V_upper) if V_upper > 0 else 0.0
        z_lower = (T_obs - E_lower) / np.sqrt(V_lower) if V_lower > 0 else 0.0

        p_upper.append(float(1 - stats.norm.cdf(z_upper)))
        p_lower.append(float(1 - stats.norm.cdf(z_lower)))

    return {
        "gamma_values": gamma_range,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "n_pairs": n_pairs,
    }


def cheatsheet() -> str:
    return "otis_sensitivity({}) -> Rosenbaum sensitivity bounds for OTIS correctional data."
