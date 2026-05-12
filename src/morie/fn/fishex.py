# morie.fn -- function file (hadesllm/morie)
"""Fisher's exact test (2x2) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import fisher_exact


def fishex(table: Union[Sequence, np.ndarray],
           alternative: str = "two-sided"):
    """Fisher's exact test for a 2x2 contingency table."""
    from ._richresult import hypothesis_test_result
    t = np.asarray(table, dtype=int)
    if t.shape != (2, 2):
        raise ValueError(f"table must be 2x2, got {t.shape}.")
    if np.any(t < 0):
        raise ValueError("cell counts must be non-negative.")
    odds, p = fisher_exact(t.tolist(), alternative=alternative)
    n = t.sum()
    warnings = []
    if n > 1000:
        warnings.append(f"n={n} is large; chi-squared would be near-equivalent and faster.")
    if np.any(t == 0):
        warnings.append("table has zero cells; OR may be 0 or infinity.")
    return hypothesis_test_result(
        test_name="Fisher exact test (2x2)",
        statistic=float(odds), pvalue=float(p),
        extra_summary=[("Odds ratio", float(odds)), ("Total n", int(n)),
                       ("Cell a", int(t[0,0])), ("Cell b", int(t[0,1])),
                       ("Cell c", int(t[1,0])), ("Cell d", int(t[1,1])),
                       ("Alternative", alternative)],
        warnings=warnings,
    )
