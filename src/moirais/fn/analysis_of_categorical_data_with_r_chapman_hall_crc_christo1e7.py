"""ContingencyTables equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_7(x):
    """
    ContingencyTables equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).

    Formula: ni+n+j/n for the one multinomial model, or equivalentlyni+ˆπ+j = ni+n+j/n under the

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
        See ``moirais.fn.describe('analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e7')`` for the full guide.

    References
    ----------
    Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ), ch.1 eq.1.7
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="ContingencyTables equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "ContingencyTables equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."},
    )


def cheatsheet():
    return "analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e7: ContingencyTables equation extracted from Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M )."
