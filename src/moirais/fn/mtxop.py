# moirais.fn — function file (hadesllm/moirais)
"""You have power over your mind — not outside events. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matrix_function(
    A: np.ndarray,
    *,
    func: str = "exp",
) -> DescriptiveResult:
    """Compute a matrix function f(A) via eigendecomposition.

    For a diagonalisable matrix A = V diag(lambda) V^{-1}, computes
    f(A) = V diag(f(lambda)) V^{-1}.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
    func : str
        Function to apply: ``"exp"``, ``"log"``, ``"sqrt"``, ``"inv"``,
        ``"abs"``, ``"sin"``, ``"cos"``.

    Returns
    -------
    DescriptiveResult
        ``value`` is the result matrix (as nested list); ``extra`` has
        eigenvalues and condition number.
    """
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    funcs = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "inv": lambda x: 1.0 / x,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
    }
    if func not in funcs:
        raise ValueError(f"func must be one of {list(funcs)}")

    eigvals, V = np.linalg.eig(A)
    cond = np.linalg.cond(V)
    f_eigvals = funcs[func](eigvals)
    result = V @ np.diag(f_eigvals) @ np.linalg.inv(V)

    if np.allclose(result.imag, 0):
        result = result.real

    return DescriptiveResult(
        name=f"Matrix {func}(A)",
        value=result.tolist(),
        extra={
            "eigenvalues": eigvals.tolist(),
            "func": func,
            "shape": list(A.shape),
            "condition_number": float(cond),
        },
    )


mtxop = matrix_function


def cheatsheet() -> str:
    return "You have power over your mind — not outside events. — Marcus Aurelius"
