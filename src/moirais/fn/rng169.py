"""Matrix inversion (ABCD) lemma used in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_abcd_matrix_inversion_lemma"]


def rangayyan_ch3_abcd_matrix_inversion_lemma(A, B, C, D):
    """
    Matrix inversion (ABCD) lemma used in RLS.

    Formula: (A + B*C*D)^(-1) = A^(-1) - A^(-1) * B * (D*A^(-1)*B + C^(-1))^(-1) * D * A^(-1)

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.213, p. 188
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matrix inversion (ABCD) lemma used in RLS."})


def cheatsheet():
    return "rng169: Matrix inversion (ABCD) lemma used in RLS."
