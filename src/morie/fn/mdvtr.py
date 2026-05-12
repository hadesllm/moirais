# morie.fn — function file (hadesllm/morie)
"""Median voter theorem (Armstrong et al. Ch 2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["median_voter", "mdvtr"]


def median_voter(x):
    """Median voter theorem (Black 1948; Armstrong et al. 2014, Ch 2).

    On a single dimension with single-peaked preferences, the Condorcet
    winner among any odd-sized electorate is the *median* of ideal
    points — NOT the mean. Returns the median plus a bootstrap-free
    Laplace-based standard error 1.2533 * s/sqrt(n) for the sample
    median of an approximately-normal sample.

    Formula: x* = median(x_i*)

    Parameters
    ----------
    x : array-like
        Voter ideal points on a single dimension.

    Returns
    -------
    RichResult with keys: estimate, se, ci_lower, ci_upper, n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": 0,
                                   "method": "median_voter"})
    est = float(np.median(x))
    if n > 1:
        # Laplace asymptotic SE of the sample median for normal-like data:
        # sqrt(pi/2) * sigma / sqrt(n) ~= 1.2533 sigma/sqrt(n).
        se = float(1.2533141373 * np.std(x, ddof=1) / np.sqrt(n))
    else:
        se = np.nan
    ci_lo = est - 1.96 * se if np.isfinite(se) else np.nan
    ci_hi = est + 1.96 * se if np.isfinite(se) else np.nan
    return RichResult(
        title="Median voter theorem (Black 1948)",
        summary_lines=[("Median (x*)", est), ("SE (Laplace)", se), ("n", n)],
        interpretation=(
            f"With single-peaked preferences in 1D, the Condorcet winner "
            f"is x* = {est:.4f} (the median ideal point)."),
        payload={"estimate": est, "se": se, "ci_lower": ci_lo,
                 "ci_upper": ci_hi, "n": n,
                 "method": "Median voter theorem"},
    )


mdvtr = median_voter


def cheatsheet():
    return "mdvtr: Median voter theorem — x* = median(x_i*) in 1D."


# CANONICAL TEST
# >>> r = median_voter([1.0, 2.0, 3.0, 4.0, 100.0])
# >>> assert abs(r["estimate"] - 3.0) < 1e-9
