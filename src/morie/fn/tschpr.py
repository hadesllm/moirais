# morie.fn -- function file (hadesllm/morie)
"""Tschuprow's T with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2_contingency


def tschpr(table: Union[Sequence, np.ndarray]):
    """Tschuprow's T: chi^2-based categorical association."""
    from ._richresult import RichResult
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError(f"table must be at least 2x2, got {t.shape}.")
    chi2, p, dof, expected = chi2_contingency(t)
    n = t.sum()
    r, c = t.shape
    T = float((chi2 / (n * ((r - 1) * (c - 1)) ** 0.5)) ** 0.5)
    if T < 0.1: bench = "negligible"
    elif T < 0.3: bench = "weak"
    elif T < 0.5: bench = "moderate"
    else: bench = "strong"
    return RichResult(
        title="Tschuprow's T (categorical association)",
        summary_lines=[
            ("Tschuprow's T", T),
            ("Strength", bench),
            ("chi^2 statistic", float(chi2)),
            ("p-value", float(p)),
            ("df", int(dof)),
            ("n total", int(n)),
            ("Table shape", t.shape),
        ],
        interpretation=(f"T={T:.3f} -> {bench} association. Range [0,1]; like "
                        "Cramer's V but uses sqrt((r-1)(c-1)) denominator."),
        payload={"value": T, "statistic": T, "chi2": float(chi2),
                 "pvalue": float(p), "dof": int(dof)},
    )
