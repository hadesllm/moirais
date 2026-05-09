# moirais.fn — function file (hadesllm/moirais)
"""2x2 crossover trial analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def crossover_analysis(
    period1: np.ndarray | list,
    period2: np.ndarray | list,
) -> DescriptiveResult:
    """
    Analyse a 2x2 crossover trial (AB/BA design).

    Computes treatment effect, period effect, and carryover test.

    Parameters
    ----------
    period1 : array-like
        Outcomes in period 1 for all subjects.
    period2 : array-like
        Outcomes in period 2 for all subjects.

    Returns
    -------
    DescriptiveResult
        extra has 'treatment_effect', 'period_effect', 'carryover_p',
        'se_treatment'.

    References
    ----------
    Senn, S. (2002). *Cross-over Trials in Clinical Research*, 2nd ed.
    Wiley, Ch. 3.
    """
    p1 = np.asarray(period1, dtype=float)
    p2 = np.asarray(period2, dtype=float)
    if len(p1) != len(p2):
        raise ValueError("Period arrays must have same length.")
    n = len(p1)
    if n < 4:
        raise ValueError("Need at least 4 subjects.")

    d = p1 - p2
    s = p1 + p2

    treatment_effect = float(np.mean(d))
    se_treat = float(np.std(d, ddof=1) / np.sqrt(n))
    t_treat = treatment_effect / se_treat if se_treat > 0 else 0
    p_treat = float(2 * stats.t.sf(abs(t_treat), df=n - 1))

    period_effect = float(np.mean(s) - np.mean(p1) - np.mean(p2))

    carryover = float(np.mean(s))
    se_carry = float(np.std(s, ddof=1) / np.sqrt(n))
    t_carry = carryover / se_carry if se_carry > 0 else 0
    p_carry = float(2 * stats.t.sf(abs(t_carry), df=n - 1))

    return DescriptiveResult(
        name="crossover_2x2",
        value=treatment_effect,
        extra={
            "treatment_effect": treatment_effect,
            "se_treatment": se_treat,
            "treatment_p": p_treat,
            "period_effect": period_effect,
            "carryover_p": p_carry,
            "n": n,
        },
    )


cross = crossover_analysis


def cheatsheet() -> str:
    return "crossover_analysis({}) -> 2x2 crossover trial analysis."
