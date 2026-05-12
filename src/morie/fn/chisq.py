# morie.fn -- function file (hadesllm/morie)
"""Chi-square goodness-of-fit or independence test."""

from typing import Union

import numpy as np
import scipy.stats as stats


def chi_square_test(
    observed: Union[list, np.ndarray],
    *,
    expected: Union[list, np.ndarray] | None = None,
) -> dict:
    """
    Chi-square goodness-of-fit (or independence) test.

    For a 1-D ``observed`` array, performs goodness-of-fit against ``expected``
    (uniform if None). For a 2-D array, performs a chi-square test of
    independence (``expected`` is ignored).

    :param observed: Observed frequencies. 1-D or 2-D array-like.
    :param expected: Expected frequencies for 1-D case. Must sum to same total
        as observed. Uniform if None.
    :return: dict with keys ``chi2``, ``df``, ``p_value``, ``expected``.
    :raises ValueError: If observed contains negative values.

    References
    ----------
    Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley. (Chapter 1.)
    """
    obs = np.asarray(observed, dtype=float)
    if np.any(obs < 0):
        raise ValueError("Observed frequencies must be non-negative.")
    if obs.ndim == 2:
        chi2_stat, p_val, df, exp = stats.chi2_contingency(obs)
        return {
            "chi2": float(chi2_stat),
            "df": int(df),
            "p_value": float(p_val),
            "expected": exp,
            "method": "Chi-square test of independence",
        }
    # 1-D goodness-of-fit
    exp_arr = np.ones_like(obs) * obs.sum() / len(obs) if expected is None else np.asarray(expected, dtype=float)
    chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp_arr)
    df = len(obs) - 1
    return {
        "chi2": float(chi2_stat),
        "df": df,
        "p_value": float(p_val),
        "expected": exp_arr,
        "method": "Chi-square goodness-of-fit test",
    }


chisq = chi_square_test


def cheatsheet() -> str:
    return "chi_square_test({}) -> Chi-square goodness-of-fit or independence test."
