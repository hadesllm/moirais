# moirais.fn — function file (hadesllm/moirais)
"""Jarque-Bera normality test. 'How wude!' -- Jar Jar Binks"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import TestResult
from ._helpers import _extract_col


def jarque_bera(data, *, col: str = "x") -> TestResult:
    """Jarque-Bera test for normality. Tests if skewness=0 and kurtosis=3.

    :param data: array-like or DataFrame.
    :param col: column name if data is a DataFrame.
    :return: TestResult with chi-squared statistic and p-value (df=2).
    """
    x = _extract_col(data, col) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float).ravel()
    x = x[np.isfinite(x)]
    stat_val, p_val = stats.jarque_bera(x)
    return TestResult(
        test_name="Jarque-Bera",
        statistic=float(stat_val),
        p_value=float(p_val),
        df=2,
        n=len(x),
        method="Jarque-Bera test for normality",
    )


jarjar = jarque_bera


def cheatsheet() -> str:
    return "jarque_bera({}) -> Jarque-Bera normality test. 'How wude!' -- Jar Jar Binks"
