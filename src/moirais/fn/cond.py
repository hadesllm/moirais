# moirais.fn — function file (hadesllm/moirais)
"""Matrix condition number."""

import numpy as np

from ._containers import ESRes


def condition_number(matrix: np.ndarray, p: int | float | str = 2) -> ESRes:
    """
    Compute the condition number of a matrix.

    :param matrix: (m, n) matrix.
    :param p: Norm type (1, 2, inf, or 'fro').
    :return: ESRes with condition number.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations. 4th ed.
    Johns Hopkins University Press.
    """
    A = np.asarray(matrix, dtype=np.float64)
    if p == "inf":
        p = np.inf
    kappa = float(np.linalg.cond(A, p=p))
    log_kappa = float(np.log10(kappa)) if kappa > 0 and np.isfinite(kappa) else float("inf")
    return ESRes(
        measure="condition_number",
        estimate=kappa,
        extra={"log10_condition": log_kappa, "p": str(p), "shape": A.shape},
    )


cond = condition_number


def cheatsheet() -> str:
    return "condition_number({}) -> Matrix condition number."
