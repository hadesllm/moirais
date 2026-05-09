# moirais.fn — function file (hadesllm/moirais)
"""Restricted cubic spline basis for flexible regression."""

import numpy as np

from ._containers import DescriptiveResult


def restricted_cubic_spline(x, knots=None, n_knots=5):
    """
    Generate restricted cubic spline (natural spline) basis matrix.

    Creates basis functions for flexible dose-response or exposure-response
    modeling with linearity constraints in the tails.

    :param x: (n,) predictor values.
    :param knots: Knot positions. If None, uses quantile-based placement.
    :param n_knots: Number of knots if knots not specified.
    :return: DescriptiveResult with basis matrix and knot positions.

    References
    ----------
    Harrell FE (2015). Regression Modeling Strategies, 2nd ed. Springer.
    Durrleman S & Simon R (1989). Flexible Regression Models with
    Cubic Splines. Stat Med 8(5):551-561.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)

    if knots is None:
        if n_knots == 3:
            pcts = [10, 50, 90]
        elif n_knots == 4:
            pcts = [5, 35, 65, 95]
        elif n_knots == 5:
            pcts = [5, 27.5, 50, 72.5, 95]
        elif n_knots == 6:
            pcts = [5, 23, 41, 59, 77, 95]
        else:
            pcts = np.linspace(5, 95, n_knots).tolist()
        knots = np.percentile(arr, pcts)
    else:
        knots = np.asarray(knots, dtype=np.float64).ravel()

    k = len(knots)
    basis = np.zeros((n, k - 2))

    def _h(u, t):
        return np.maximum(u - t, 0) ** 3

    tk = knots[-1]
    tkm1 = knots[-2]
    denom = tk - tkm1

    for j in range(k - 2):
        tj = knots[j]
        basis[:, j] = (_h(arr, tj) - _h(arr, tkm1) * (tk - tj) / denom + _h(arr, tk) * (tkm1 - tj) / denom) / (
            tk - knots[0]
        ) ** 2

    return DescriptiveResult(
        name="restricted_cubic_spline",
        value=float(k),
        extra={
            "basis_matrix": basis.tolist(),
            "knots": knots.tolist(),
            "n_knots": int(k),
            "n_basis_functions": int(k - 2),
            "n_observations": n,
        },
    )


def cheatsheet() -> str:
    return "restricted_cubic_spline({}) -> Restricted cubic spline basis for flexible regression."
