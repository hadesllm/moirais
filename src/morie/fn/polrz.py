# morie.fn — function file (hadesllm/morie)
"""Political polarization index (Armstrong Ch 8)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["polarization_index", "polrz"]


def polarization_index(x, group=None):
    """Two-group polarization: P = |mean(x_R) - mean(x_D)| / pooled_sd.

    A Cohen-d-style standardised distance between two party-aligned
    distributions of ideal points (Armstrong et al. 2014, Ch 8).

    Parameters
    ----------
    x : array-like (n,)
        Ideal points / scores for each legislator.
    group : array-like (n,), optional
        Two-level group indicator (any hashable). If None, splits at
        the median (right of median = group 1).

    Returns
    -------
    RichResult with keys: estimate, mean_R, mean_D, sd_R, sd_D,
        pooled_sd, n_R, n_D
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "polarization_index"})
    if group is None:
        med = np.median(x)
        g = (x >= med).astype(int)
    else:
        g_arr = np.asarray(group).ravel()
        if g_arr.size != n:
            raise ValueError("group must have the same length as x")
        uniq = list({v for v in g_arr})
        if len(uniq) != 2:
            raise ValueError("group must have exactly 2 levels")
        g = (g_arr == uniq[1]).astype(int)
    xR = x[g == 1]; xD = x[g == 0]
    if xR.size < 1 or xD.size < 1:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "polarization_index"})
    mR, mD = float(xR.mean()), float(xD.mean())
    sR = float(xR.std(ddof=1)) if xR.size > 1 else 0.0
    sD = float(xD.std(ddof=1)) if xD.size > 1 else 0.0
    # Pooled SD using the standard Cohen pooled formula
    nR, nD = xR.size, xD.size
    pooled = np.sqrt(((nR - 1) * sR ** 2 + (nD - 1) * sD ** 2)
                     / max(nR + nD - 2, 1))
    # If both groups are degenerate (constant within-group), fall back to
    # the overall sample SD so that perfectly bimodal inputs still yield a
    # finite polarization score (the Cohen-d limit, +Inf, is reported as
    # the overall-SD-normalised distance).
    if pooled <= 0:
        pooled = float(x.std(ddof=1)) if x.size > 1 else 0.0
    pol = abs(mR - mD) / pooled if pooled > 0 else np.nan
    return RichResult(
        title="Polarization index (Cohen-d on ideal points)",
        summary_lines=[("|mean(R) - mean(D)|", abs(mR - mD)),
                       ("pooled SD", float(pooled)),
                       ("Polarization P", float(pol) if np.isfinite(pol)
                                          else pol),
                       ("n_R / n_D", f"{nR}/{nD}")],
        interpretation=(
            f"P = {pol:.4f}; values > 0.8 indicate substantial "
            f"between-party divergence (Cohen 1988 large-effect cutoff)."),
        payload={"estimate": float(pol) if np.isfinite(pol) else np.nan,
                 "mean_R": mR, "mean_D": mD, "sd_R": sR, "sd_D": sD,
                 "pooled_sd": float(pooled), "n_R": int(nR), "n_D": int(nD),
                 "method": "polarization_index"},
    )


polrz = polarization_index


def cheatsheet():
    return "polrz: Polarization = |mean(R)-mean(D)|/pooled_sd."


# CANONICAL TEST
# >>> r = polarization_index([-1,-1,-1,1,1,1])
# >>> assert r["estimate"] > 1.0
