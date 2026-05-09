# moirais.fn — function file (hadesllm/moirais)
"""Anderson-Darling test for Normality with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import anderson


def anddrl(x: Union[Sequence, np.ndarray]):
    """Anderson-Darling test for Normality (more tail-sensitive than Shapiro)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 8:
        raise ValueError(f"need at least 8 observations, got {a.size}.")
    res = anderson(a, dist="norm")
    A2 = float(res.statistic)
    crit_rows = []
    for sig, c in zip(res.significance_level, res.critical_values):
        verdict = "Reject H0" if A2 > c else "Fail to reject H0"
        crit_rows.append([f"{sig:.1f}%", f"{c:.4f}", verdict])
    return RichResult(
        title="Anderson-Darling test for Normality",
        summary_lines=[("Test statistic A^2", A2), ("n", int(a.size)),
                       ("Sample mean", float(a.mean())),
                       ("Sample SD", float(a.std(ddof=1)))],
        tables=[{"title": "Critical values (right-tail):",
                 "headers": ["Significance level", "Critical value", "At this level"],
                 "rows": crit_rows}],
        warnings=[] if a.size <= 5000 else
                 [f"n={a.size} very large; tiny deviations yield large A^2."],
        interpretation="A^2 vs. critical values above gives rejection at each alpha.",
        payload={"statistic": A2,
                 "critical_values": list(res.critical_values),
                 "significance_levels": list(res.significance_level)},
    )
