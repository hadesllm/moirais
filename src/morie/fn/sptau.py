"""Spatial autocorrelation (Moran I/Geary C unified)."""
import numpy as np
from scipy import stats as _scistats
from ._richresult import RichResult

__all__ = ["spatial_autocorrelation"]


def spatial_autocorrelation(x, w):
    """
    Spatial autocorrelation (Moran I/Geary C unified).

    Formula: I = n/S0 * sum_ij w_ij(x_i-xbar)(x_j-xbar) / sum(x_i-xbar)^2
    where S0 = sum_ij w_ij.

    Inference uses the normal approximation with the standard randomization
    variance for Moran's I (Cliff & Ord 1981; Schabenberger & Gotway 2005, Ch 1).

    Parameters
    ----------
    x : array-like, shape (n,)
        Observed values at the n spatial units.
    w : array-like, shape (n, n)
        Spatial weights matrix (row-stochastic or binary contiguity).

    Returns
    -------
    RichResult with payload:
        statistic : Moran's I
        p_value   : two-sided normal-approximation p-value
        n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    W = np.asarray(w, dtype=float)
    n = x.size
    if W.shape != (n, n):
        raise ValueError(
            f"w must be a square matrix matching x; got {W.shape} vs n={n}"
        )
    if n < 3:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "n": n,
            "method": "Moran's I (spatial autocorrelation)",
        })
    xbar = x.mean()
    z = x - xbar
    S0 = W.sum()
    if S0 == 0 or np.dot(z, z) == 0:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "n": n,
            "method": "Moran's I (spatial autocorrelation)",
        })
    num = z @ W @ z
    den = np.dot(z, z)
    I = (n / S0) * (num / den)

    # Expectation and variance under randomization (Cliff & Ord 1981).
    EI = -1.0 / (n - 1)
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    m2 = (z ** 2).mean()
    m4 = (z ** 4).mean()
    b2 = m4 / (m2 ** 2) if m2 > 0 else 3.0
    A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    if C <= 0:
        var_I = float("nan")
        zscore = float("nan")
        p_value = float("nan")
    else:
        var_I = (A - B) / C - EI ** 2
        if not np.isfinite(var_I) or var_I <= 0:
            zscore = float("nan")
            p_value = float("nan")
        else:
            zscore = (I - EI) / np.sqrt(var_I)
            p_value = 2.0 * (1.0 - _scistats.norm.cdf(abs(zscore)))

    return RichResult(payload={
        "statistic": float(I),
        "p_value": float(p_value),
        "expectation": float(EI),
        "variance": float(var_I) if np.isfinite(var_I) else float("nan"),
        "z_score": float(zscore) if np.isfinite(zscore) else float("nan"),
        "n": int(n),
        "method": "Moran's I (spatial autocorrelation)",
    })


def cheatsheet():
    return "sptau: Spatial autocorrelation (Moran I)"


# CANONICAL TEST
# x = [1.0, 2.0, 3.0, 4.0, 5.0]
# W = 5x5 path-graph contiguity: W[i,j]=1 iff |i-j|==1, else 0  (S0 = 8)
# Expected:  I = (5/8) * ( sum z_i z_{i+1} + sum z_{i+1} z_i ) / sum z_i^2
#          ~ 0.4
