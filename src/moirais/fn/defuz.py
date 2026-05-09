# moirais.fn — function file (hadesllm/moirais)
"""Defuzzification methods."""

import numpy as np

from ._containers import DescriptiveResult
def defuzzify(x, mf, method: str = "centroid", **kwargs) -> DescriptiveResult:
    """
    Defuzzify a fuzzy set to obtain a crisp output value.

    Supported methods:

    - **centroid** (centre of gravity):
      :math:`z^* = \\frac{\\int x \\cdot \\mu(x)\\,dx}{\\int \\mu(x)\\,dx}`
    - **bisector**: x-value that splits the area in half
    - **mom** (mean of maximum): mean of x where :math:`\\mu(x)` is maximal
    - **som** (smallest of maximum): smallest x with maximum membership
    - **lom** (largest of maximum): largest x with maximum membership

    :param x: Universe of discourse (sorted array).
    :param mf: Membership function values (same length as x).
    :param method: Defuzzification method. Default ``"centroid"``.
    :return: DescriptiveResult with crisp output value.
    :raises ValueError: If method is unknown or membership sums to zero.

    References
    ----------
    Mamdani, E. H. & Assilian, S. (1975). An experiment in linguistic
    synthesis with a fuzzy logic controller. *International Journal of
    Man-Machine Studies*, 7(1), 1-13.
    """
    x = np.asarray(x, dtype=np.float64)
    mf = np.asarray(mf, dtype=np.float64)

    if len(x) != len(mf):
        raise ValueError("x and mf must have the same length.")
    if np.sum(mf) < 1e-300:
        raise ValueError("Membership function sums to zero; cannot defuzzify.")

    if method == "centroid":
        z_star = float(np.trapezoid(x * mf, x) / np.trapezoid(mf, x))
    elif method == "bisector":
        cumarea = np.cumsum(mf[:-1] * np.diff(x))
        total = cumarea[-1]
        idx = int(np.searchsorted(cumarea, total / 2.0))
        z_star = float(x[min(idx, len(x) - 1)])
    elif method == "mom":
        max_val = np.max(mf)
        max_mask = np.isclose(mf, max_val)
        z_star = float(np.mean(x[max_mask]))
    elif method == "som":
        max_val = np.max(mf)
        max_mask = np.isclose(mf, max_val)
        z_star = float(x[max_mask][0])
    elif method == "lom":
        max_val = np.max(mf)
        max_mask = np.isclose(mf, max_val)
        z_star = float(x[max_mask][-1])
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'centroid', 'bisector', 'mom', 'som', or 'lom'.")

    return DescriptiveResult(
        name="defuzzify",
        value=z_star,
        extra={
            "crisp_output": z_star,
            "method": method,
            "total_area": float(np.trapezoid(mf, x)),
            "max_membership": float(np.max(mf)),
        },
    )


defuz = defuzzify


def cheatsheet() -> str:
    return "defuzzify({}) -> Defuzzification methods."
