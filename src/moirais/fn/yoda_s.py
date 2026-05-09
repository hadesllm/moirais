"""Sensitivity analysis. 'Numbers have life; they're not just symbols on paper. — Shakuntala Devi'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def sensitivity_analysis(
    ate: float,
    se: float,
    *,
    gamma_range: tuple = (1.0, 3.0),
    n_gamma: int = 10,
) -> DescriptiveResult:
    """Rosenbaum-style sensitivity: how much hidden bias (Gamma) would overturn the result?"""
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    results = []
    for g in gammas:
        # Under Gamma, the test statistic bounds shift
        t_upper = ate / se + np.log(g)
        t_lower = ate / se - np.log(g)
        p_upper = float(2 * stats.norm.sf(abs(t_lower)))
        results.append({"gamma": float(g), "p_upper": p_upper, "significant_5pct": p_upper < 0.05})

    # Find critical gamma where result becomes non-significant
    critical = next((r["gamma"] for r in results if not r["significant_5pct"]), gammas[-1])
    return DescriptiveResult(
        name="Rosenbaum sensitivity",
        value=critical,
        extra={"table": results, "critical_gamma": critical},
    )


yoda_s = sensitivity_analysis


def cheatsheet() -> str:
    return "sensitivity_analysis({}) -> Sensitivity analysis. 'Numbers have life; they're not just symbols on paper. — Shakuntala Devi' -- Yo"
