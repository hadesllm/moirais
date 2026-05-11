# morie.fn — function file (hadesllm/morie)
"""Fuzzy AND (t-norm) operation."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The whole is greater than the sum of its parts. — Aristotle"


def fuzzy_and(a, b, method: str = "min", **kwargs) -> DescriptiveResult:
    """
    Compute fuzzy AND (t-norm) of two membership values.

    Supported t-norms:

    - **min**: :math:`T(a, b) = \\min(a, b)` (Zadeh)
    - **product**: :math:`T(a, b) = a \\cdot b` (algebraic product)
    - **lukasiewicz**: :math:`T(a, b) = \\max(0, a + b - 1)`
    - **drastic**: :math:`T(a, b) = \\min(a, b)` if :math:`\\max(a,b)=1`, else 0

    :param a: Membership value(s) in [0, 1].
    :param b: Membership value(s) in [0, 1].
    :param method: T-norm type. Default ``"min"``.
    :return: DescriptiveResult with t-norm result.
    :raises ValueError: If method is unknown.

    References
    ----------
    Klement, E. P., Mesiar, R. & Pap, E. (2000). *Triangular Norms*.
    Springer. doi:10.1007/978-94-015-9540-7
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if method == "min":
        result = np.minimum(a, b)
    elif method == "product":
        result = a * b
    elif method == "lukasiewicz":
        result = np.maximum(0.0, a + b - 1.0)
    elif method == "drastic":
        result = np.where(
            np.maximum(a, b) == 1.0,
            np.minimum(a, b),
            0.0,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'min', 'product', 'lukasiewicz', or 'drastic'.")

    return DescriptiveResult(
        name="fuzzy_and",
        value=float(np.mean(result)),
        extra={
            "result": result,
            "method": method,
            "mean": float(np.mean(result)),
        },
    )


fzand = fuzzy_and


def cheatsheet() -> str:
    return "fuzzy_and({}) -> Fuzzy AND (t-norm) operation."
