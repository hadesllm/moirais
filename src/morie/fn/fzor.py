# morie.fn — function file (hadesllm/morie)
"""Fuzzy OR (t-conorm) operation."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this. -- Everyone"


def fuzzy_or(a, b, method: str = "max", **kwargs) -> DescriptiveResult:
    """
    Compute fuzzy OR (t-conorm / s-norm) of two membership values.

    Supported s-norms:

    - **max**: :math:`S(a, b) = \\max(a, b)` (Zadeh)
    - **probsum**: :math:`S(a, b) = a + b - a \\cdot b` (probabilistic sum)
    - **lukasiewicz**: :math:`S(a, b) = \\min(1, a + b)` (bounded sum)
    - **drastic**: :math:`S(a, b) = \\max(a, b)` if :math:`\\min(a,b)=0`, else 1

    :param a: Membership value(s) in [0, 1].
    :param b: Membership value(s) in [0, 1].
    :param method: S-norm type. Default ``"max"``.
    :return: DescriptiveResult with s-norm result.
    :raises ValueError: If method is unknown.

    References
    ----------
    Klement, E. P., Mesiar, R. & Pap, E. (2000). *Triangular Norms*.
    Springer. doi:10.1007/978-94-015-9540-7
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if method == "max":
        result = np.maximum(a, b)
    elif method == "probsum":
        result = a + b - a * b
    elif method == "lukasiewicz":
        result = np.minimum(1.0, a + b)
    elif method == "drastic":
        result = np.where(
            np.minimum(a, b) == 0.0,
            np.maximum(a, b),
            1.0,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'max', 'probsum', 'lukasiewicz', or 'drastic'.")

    return DescriptiveResult(
        name="fuzzy_or",
        value=float(np.mean(result)),
        extra={
            "result": result,
            "method": method,
            "mean": float(np.mean(result)),
        },
    )


fzor = fuzzy_or


def cheatsheet() -> str:
    return "fuzzy_or({}) -> Fuzzy OR (t-conorm) operation."
