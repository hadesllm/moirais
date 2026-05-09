# moirais.fn — function file (hadesllm/moirais)
"""Breusch-Pagan heteroscedasticity test with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2


def bptest(residuals: Union[Sequence, np.ndarray],
           X: Union[Sequence, np.ndarray]):
    """Breusch-Pagan test - is residual variance a function of X?"""
    from ._richresult import hypothesis_test_result
    e = np.asarray(residuals, dtype=float); X = np.asarray(X, dtype=float)
    n, p = X.shape
    if e.size != n:
        raise ValueError(f"residuals length ({e.size}) != X rows ({n}).")
    e2 = e * e
    Xc = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(Xc, e2, rcond=None)
    yhat = Xc @ beta
    ss_tot = np.sum((e2 - e2.mean()) ** 2)
    ss_res = np.sum((e2 - yhat) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = float(n * r2)
    return hypothesis_test_result(
        test_name="Breusch-Pagan test for heteroscedasticity",
        statistic=lm, df=p, pvalue=float(1 - chi2.cdf(lm, p)),
        extra_summary=[
            ("R^2 of e^2 on X", r2), ("LM = n * R^2", lm),
            ("n observations", n), ("Predictors in X", p),
        ],
    )
