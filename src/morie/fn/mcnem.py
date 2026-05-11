# morie.fn — function file (hadesllm/morie)
"""McNemar test (paired binary 2x2) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2


def mcnem(table: Union[Sequence, np.ndarray], continuity: bool = True):
    """McNemar test for paired binary outcomes (2x2)."""
    from ._richresult import hypothesis_test_result
    t = np.asarray(table, dtype=float)
    if t.shape != (2, 2):
        raise ValueError(f"table must be 2x2, got {t.shape}.")
    a, b, c, d = t[0,0], t[0,1], t[1,0], t[1,1]
    warnings = []
    if b + c == 0:
        warnings.append("no discordant pairs (b+c=0); test undefined.")
        return hypothesis_test_result(
            test_name="McNemar test (paired binary)",
            statistic=0.0, pvalue=1.0, df=1,
            extra_summary=[("Concordant a", int(a)), ("Discordant b", 0),
                           ("Discordant c", 0), ("Concordant d", int(d))],
            warnings=warnings,
        )
    chi = (abs(b - c) - 0.5) ** 2 / (b + c) if continuity else (b - c) ** 2 / (b + c)
    if (b + c) < 10:
        warnings.append(f"discordant total b+c={int(b+c)}<10; consider exact mid-p binomial.")
    return hypothesis_test_result(
        test_name="McNemar test (paired binary)",
        statistic=float(chi), df=1, pvalue=float(1 - chi2.cdf(chi, 1)),
        extra_summary=[("Concordant a", int(a)), ("Discordant b", int(b)),
                       ("Discordant c", int(c)), ("Concordant d", int(d)),
                       ("Continuity correction", continuity)],
        warnings=warnings,
    )
