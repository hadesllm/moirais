# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bartlett's test of sphericity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn._containers import BrtRes


def bart(data: pd.DataFrame | np.ndarray, cdf=None) -> BrtRes:
    """Bartlett's test of sphericity.

    H0: correlation matrix = identity. Reject (p < 0.05) indicates
    the data is factorable.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    BrtRes
        Chi-square statistic, degrees of freedom, and p-value.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)

    det_R = max(np.linalg.det(R), 1e-15)
    chisq = -(n - 1 - (2 * k + 5) / 6) * np.log(det_R)
    df = k * (k - 1) // 2
    pval = 1 - sp.chi2.cdf(chisq, df)

    return BrtRes(chisq=float(chisq), df=df, pval=float(pval))


def cheatsheet() -> str:
    return "bart({}) -> Bartlett's test of sphericity."
