"""Toeplitz matrix construction."""

import numpy as np
from scipy.linalg import toeplitz as _toeplitz

from ._containers import DescriptiveResult

_QUOTE = "Patience is bitter, but its fruit is sweet. -- Aristotle"


def toeplitz_matrix(c, r=None, **kwargs) -> DescriptiveResult:
    """
    Construct a Toeplitz matrix from first column and first row.

    A Toeplitz matrix has constant diagonals; commonly used for
    representing autocorrelation matrices in signal processing.

    :param c: array-like, first column of the Toeplitz matrix.
    :param r: array-like, first row. If None, uses conjugate of c.
    :return: DescriptiveResult with matrix shape and properties.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations, 4th ed.
    Johns Hopkins University Press.
    """
    c = np.asarray(c, dtype=np.float64).ravel()
    if r is not None:
        r = np.asarray(r, dtype=np.float64).ravel()
    T = _toeplitz(c, r)
    sym = bool(np.allclose(T, T.T))
    return DescriptiveResult(
        name="toeplitz_matrix",
        value=float(T.shape[0]),
        extra={
            "shape": list(T.shape),
            "symmetric": sym,
            "trace": float(np.trace(T)),
            "frobenius_norm": float(np.linalg.norm(T, "fro")),
        },
    )


toepz = toeplitz_matrix


def cheatsheet() -> str:
    return "toeplitz_matrix({}) -> Toeplitz matrix construction."
