# moirais.fn — function file (hadesllm/moirais)
"""Wald joint test with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2


def waldjt(beta: Union[Sequence, np.ndarray],
           cov: Union[Sequence, np.ndarray],
           R: Union[Sequence, np.ndarray],
           r: Union[Sequence, np.ndarray]):
    """Joint Wald test of H0: R*beta = r."""
    from ._richresult import hypothesis_test_result
    b = np.asarray(beta, dtype=float)
    S = np.asarray(cov, dtype=float)
    R = np.asarray(R, dtype=float)
    r = np.asarray(r, dtype=float)
    diff = R @ b - r
    inner = R @ S @ R.T
    W = float(diff @ np.linalg.pinv(inner) @ diff)
    df = int(np.linalg.matrix_rank(R))
    p = float(1 - chi2.cdf(W, df))
    return hypothesis_test_result(
        test_name="Wald joint test",
        statistic=W, df=df, pvalue=p,
        extra_summary=[
            ("Number of restrictions", df),
            ("Parameters tested", b.size),
            ("R-beta - r norm", float(np.linalg.norm(diff))),
        ],
    )
