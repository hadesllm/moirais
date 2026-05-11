# morie.fn — function file (hadesllm/morie)
"""Impulse Response Function from a VAR coefficient matrix."""

import numpy as np


def irf(
    var_coefficients: np.ndarray,
    periods: int = 20,
    shock_var: int = 0,
    shock_size: float = 1.0,
) -> np.ndarray:
    """
    Impulse Response Function from a VAR(1) model.

    Computes the response of each variable to a one-unit shock in
    variable *shock_var* by recursively applying the VAR companion
    matrix:

    .. math::

        \\mathbf{y}_{t+h} = A^h \\mathbf{e}_j

    where :math:`\\mathbf{e}_j` is a unit shock to variable *j*.

    :param var_coefficients: VAR(1) coefficient matrix *A* of shape
        (k, k) where *k* is the number of endogenous variables.
    :param periods: Number of periods for the IRF. Default 20.
    :param shock_var: Index of the variable receiving the shock. Default 0.
    :param shock_size: Magnitude of the shock. Default 1.0.
    :return: ndarray of shape (periods + 1, k) where row *h* gives the
        response at horizon *h*.
    :raises ValueError: If *var_coefficients* is not square or
        *shock_var* is out of range.

    References
    ----------
    Lutkepohl, H. (2005). New Introduction to Multiple Time Series
    Analysis. Springer.

    Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*,
    48(1), 1-48.
    """
    A = np.asarray(var_coefficients, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"var_coefficients must be a square matrix, got shape {A.shape}.")
    k = A.shape[0]
    if shock_var < 0 or shock_var >= k:
        raise ValueError(f"shock_var must be in [0, {k - 1}], got {shock_var}.")
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}.")

    responses = np.zeros((periods + 1, k))
    # Initial shock
    impulse = np.zeros(k)
    impulse[shock_var] = shock_size
    responses[0] = impulse

    # Propagate
    A_power = np.eye(k)
    for h in range(1, periods + 1):
        A_power = A_power @ A
        responses[h] = A_power @ impulse

    return responses


def cheatsheet() -> str:
    return "irf({}) -> Impulse Response Function from a VAR coefficient matrix."
