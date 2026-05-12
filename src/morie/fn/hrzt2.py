# morie.fn — function file (hadesllm/morie)
"""Local Average Treatment Effect (LATE) via IV Wald estimator
(Imbens-Angrist 1994; Horowitz 2009, Ch 9).

    LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])

with delta-method SE.  Handles continuous or binary instruments by
discretising the instrument at its median; for binary Z this is the
exact Wald estimator.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_local_ate"]


def horowitz_local_ate(x, y, z, treatment):
    """IV LATE estimator (Wald form).

    Parameters
    ----------
    x : array-like or None
        Optional covariates (currently treated as informational only;
        the bare Wald estimator does not condition on X — partial-out
        could be added in future).
    y : array-like
        Outcome.
    z : array-like
        Instrument (binary or continuous; continuous is dichotomised at
        the median).
    treatment : array-like
        Treatment indicator D.
    """
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    D = np.asarray(treatment, dtype=float).ravel()
    n = y.size
    if n < 20 or z.size != n or D.size != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "LATE (insufficient data)"})
    # Dichotomise Z if non-binary
    uniq = np.unique(z)
    if uniq.size > 2:
        z_bin = (z > np.median(z)).astype(float)
    else:
        z_bin = (z == uniq.max()).astype(float)
    n1 = (z_bin > 0.5).sum(); n0 = (z_bin < 0.5).sum()
    if n1 < 5 or n0 < 5:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "LATE (one arm of Z empty)"})
    Ybar1 = y[z_bin > 0.5].mean(); Ybar0 = y[z_bin < 0.5].mean()
    Dbar1 = D[z_bin > 0.5].mean(); Dbar0 = D[z_bin < 0.5].mean()
    num = Ybar1 - Ybar0
    den = Dbar1 - Dbar0
    if abs(den) < 1e-8:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "LATE (weak instrument: D̄ does not vary with Z)"})
    late = num / den
    # Delta-method SE — treat Ȳ_z, D̄_z as 4 independent sample means
    var_y1 = y[z_bin > 0.5].var(ddof=1) / n1
    var_y0 = y[z_bin < 0.5].var(ddof=1) / n0
    var_d1 = D[z_bin > 0.5].var(ddof=1) / n1
    var_d0 = D[z_bin < 0.5].var(ddof=1) / n0
    var_num = var_y1 + var_y0
    var_den = var_d1 + var_d0
    # Var(num/den) ≈ (1/den^2) (var_num + (num/den)^2 var_den)
    var_late = (var_num + (late ** 2) * var_den) / (den ** 2)
    se = float(np.sqrt(max(var_late, 0)))
    return RichResult(payload={
        "estimate": float(late), "se": se,
        "first_stage": float(den),
        "reduced_form": float(num),
        "n": n,
        "method": "IV Wald estimator (Imbens-Angrist LATE)",
    })


def cheatsheet():
    return "hrzt2: IV LATE (Wald)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(12)
    n = 1000
    z = rng.integers(0, 2, size=n).astype(float)
    # compliers: 60% comply
    comply = rng.uniform(size=n) < 0.6
    D = np.where(comply, z, rng.integers(0, 2, size=n).astype(float))
    y = 0.5 + 2.0 * D + 0.5 * rng.standard_normal(n)
    res = horowitz_local_ate(None, y, z, D)
    print(res)
    assert abs(res["estimate"] - 2.0) < 0.3
