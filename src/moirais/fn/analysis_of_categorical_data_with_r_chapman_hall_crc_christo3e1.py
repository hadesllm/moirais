"""CountModels equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_1(x):
    """
    CountModels equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).

    Formula: n1 =w, n2 =n−w, π1 =π, andπ2 = 1−π in the notation of that section

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
        See ``moirais.fn.describe('analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e1')`` for the full guide.

    References
    ----------
    Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ), ch.3 eq.3.1
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."},
    )


def cheatsheet():
    return "analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e1: CountModels equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."
