# morie.fn -- function file (hadesllm/morie)
"""Variance Inflation Factor."""

from typing import Sequence, Union
import numpy as np
def vif(X: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Variance Inflation Factor for each column of X.

    Regress each column on the others; VIFⱼ = 1 / (1 − Rⱼ²).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out = np.zeros(p)
    for j in range(p):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        if Xo.shape[1] == 0:
            out[j] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        yhat = Xo @ beta
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - yhat) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        out[j] = 1.0 / max(1 - r2, 1e-12)
    return out


# Back-compat alias -- older imports reference `variance_inflation`.
variance_inflation = vif
