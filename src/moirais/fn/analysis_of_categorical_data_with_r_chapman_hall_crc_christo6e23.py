"""Bayesian equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23(x):
    """
    Bayesian equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).

    Formula: [EQ] n−w +b)πw+a−1(1−π)n+b−w−1 (6.23)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``moirais.fn.describe('analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e23')`` for the full guide.

    References
    ----------
    Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ), ch.6 eq.6.23
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Bayesian equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Bayesian equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."},
    )


def cheatsheet():
    return "analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e23: Bayesian equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."
